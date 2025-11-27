import os
import argparse
from dotenv import load_dotenv
import numpy as np
import pandas as pd

from CBF.CBF_recommend import generate_cbf_recommendations
from CBF.cbf_model import load_catalogue_and_features
from CF.CF_recommend import generate_cf_recommendations


# ======================================================
# CONFIG

TOP_N = 20                     # Final recommendations to show

MIN_PLAYTIME = 60              # Minimum minutes for an owned game to count strongly

# How many candidates each side (CBF / CF) contributes
CANDIDATE_POOL_SIZE = 500

# CBF internal MMR: set to 0.0 as a Disable Flag
LAMBDA_MMR_CBF = 0.0

# Final MMR on HYBRID scores
LAMBDA_MMR_HYBRID = 0.6        # relevance vs diversity

BETA_ANCHOR_BLEND = 0.3        # anchor vs global CBF in user scoring

ALPHA_HYBRID = 0.35            # CF vs CBF weight in final hybrid score


# ======================================================
# HYBRID HOOKS (CBF + CF)

def normalise_scores(scores: np.ndarray) -> np.ndarray:
    """
    Min–max normalisation to [0, 1] for a 1D score vector.

    If scores are constant, returns a zero vector to avoid NaNs.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores

    s_min = scores.min()
    s_max = scores.max()

    if s_max <= s_min:
        # All scores identical (or degenerate) → return zeros
        return np.zeros_like(scores)

    return (scores - s_min) / (s_max - s_min)


def combine_cbf_cf(
    cbf_scores: np.ndarray,
    cf_scores: np.ndarray | None = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Combine CBF and CF scores into a single hybrid score vector.

    IMPORTANT:
        Inputs are assumed to ALREADY be in a comparable scale,
        typically [0, 1]. No normalisation is done here.

    Hybrid(u, i) = alpha * CF(u, i) + (1 - alpha) * CBF(u, i)

    If cf_scores is None, this function just returns cbf_scores.
    """
    cbf_scores = np.asarray(cbf_scores, dtype=float)

    if cf_scores is None:
        return cbf_scores

    cf_scores = np.asarray(cf_scores, dtype=float)
    if cf_scores.shape != cbf_scores.shape:
        raise ValueError(
            f"Shape mismatch: cbf_scores {cbf_scores.shape}, cf_scores {cf_scores.shape}"
        )

    alpha = float(alpha)
    alpha = max(0.0, min(1.0, alpha))  # clamp to [0, 1]

    return alpha * cf_scores + (1.0 - alpha) * cbf_scores


# ======================================================
# MMR ON HYBRID

def mmr_rerank(
    appids: np.ndarray,
    hybrid_scores: np.ndarray,
    lambda_mmr: float,
    top_k: int,
) -> np.ndarray:
    """
    Apply MMR on the hybrid scores, using TF–IDF content similarity
    from the CBF feature matrix.

    Parameters
    ----------
    appids : np.ndarray[int]
        Candidate appids.
    hybrid_scores : np.ndarray[float]
        Relevance scores to use in MMR.
    lambda_mmr : float
        Relevance vs diversity trade-off.
    top_k : int
        Number of items to select.

    Returns
    -------
    selected_indices : np.ndarray[int]
        Indices into the original appids array in the MMR order.
    """
    appids = np.asarray(appids, dtype=int)
    hybrid_scores = np.asarray(hybrid_scores, dtype=float)

    if appids.size == 0:
        return np.array([], dtype=int)

    # Load catalogue + feature matrix (L2-normalised)
    catalogue_df, full_matrix_norm = load_catalogue_and_features()
    appid_to_idx = {
        int(a): i
        for i, a in enumerate(catalogue_df["appid"].astype(int).to_numpy())
    }

    # Map candidate appids to rows in the feature matrix
    row_indices = []
    for appid in appids:
        idx = appid_to_idx.get(int(appid), None)
        row_indices.append(-1 if idx is None else idx)

    row_indices = np.asarray(row_indices, dtype=int)
    valid_mask = row_indices >= 0

    if not np.any(valid_mask):
        # No content vectors → fallback to pure relevance ranking
        order = np.argsort(-hybrid_scores)
        return order[:top_k]

    valid_positions = np.where(valid_mask)[0]
    mat_rows = row_indices[valid_positions]

    # Candidate vectors (assumed L2-normalised)
    cand_vecs = full_matrix_norm[mat_rows]  # shape (M, D)

    # Precompute similarity matrix in the valid subspace
    # cand_vecs is sparse CSR; convert the result to a dense array
    sim_matrix_sparse = cand_vecs @ cand_vecs.T
    sim_matrix = sim_matrix_sparse.toarray()  # shape (M, M) dense ndarray

    r_valid = hybrid_scores[valid_positions]
    selected_valid: list[int] = []

    max_items = min(top_k, len(valid_positions))

    while len(selected_valid) < max_items:
        if not selected_valid:
            # First item: highest relevance
            best_local = int(np.argmax(r_valid))
        else:
            selected_arr = np.array(selected_valid, dtype=int)

            # max similarity to any already selected item
            # sim_matrix[:, selected_arr] -> (M, len(selected))
            # max(..., axis=1) -> (M, 1) so we flatten it
            max_sims = sim_matrix[:, selected_arr].max(axis=1)
            max_sims = np.asarray(max_sims).ravel()  # (M,)

            mmr_scores = (
                lambda_mmr * r_valid
                - (1.0 - lambda_mmr) * max_sims
            )
            # mask already selected
            mmr_scores[selected_arr] = -np.inf
            best_local = int(np.argmax(mmr_scores))
            if np.isneginf(mmr_scores[best_local]):
                break

        selected_valid.append(best_local)

    # Map back to global positions in appids
    selected_global = [valid_positions[i] for i in selected_valid]

    # If fewer than top_k, fill with remaining by pure relevance
    if len(selected_global) < min(top_k, len(appids)):
        remaining = [i for i in range(len(appids)) if i not in selected_global]
        remaining_sorted = sorted(
            remaining,
            key=lambda idx: hybrid_scores[idx],
            reverse=True,
        )
        need = min(top_k, len(appids)) - len(selected_global)
        selected_global.extend(remaining_sorted[:need])

    return np.array(selected_global, dtype=int)


# ======================================================
# MAIN

def main(steamid64: str):
    load_dotenv()
    api_key = os.getenv("STEAM_API_KEY")
    if not api_key:
        raise RuntimeError("STEAM_API_KEY not found in .env")

    steamid64 = steamid64.strip()
    if not steamid64:
        raise RuntimeError("SteamID64 is required")

    # --------------------------------------------------
    # 1) CBF: get its own top-K candidates (no diversity here)
    cbf_recs = generate_cbf_recommendations(
        steamid64=steamid64,
        api_key=api_key,
        top_n=CANDIDATE_POOL_SIZE,
        min_playtime=MIN_PLAYTIME,
        candidate_pool_size=CANDIDATE_POOL_SIZE,
        beta_anchor_blend=BETA_ANCHOR_BLEND,
        lambda_mmr=LAMBDA_MMR_CBF,  # 0.0 → pure relevance list (internal MMR off)
    )

    # ---------- NEW: pure CF fallback if CBF fails ----------
    if cbf_recs.empty:
        print("\n[HYBRID] No content-based recommendations generated; falling back to pure CF.")
        cf_recs = generate_cf_recommendations(
            steamid64=steamid64,
            top_k=TOP_N,
        )

        if cf_recs is None or cf_recs.empty:
            print("\n[CF] No CF recommendations available for this user either.")
            return

        # Normalise CF scores for readability
        cf_recs = cf_recs.copy()
        cf_recs["cf_score_norm"] = normalise_scores(
            cf_recs["cf_score_raw"].to_numpy(dtype=float)
        )

        # Try to attach titles from the CBF catalogue
        try:
            catalogue_df, _ = load_catalogue_and_features()
            appid_to_title = {
                int(a): t
                for a, t in zip(
                    catalogue_df["appid"].astype(int).to_numpy(),
                    catalogue_df["title"].astype(str).to_numpy(),
                )
            }
        except Exception:
            appid_to_title = {}

        print("\nTop Recommendations (Pure CF fallback):")
        # Use raw CF ranking but only show TOP_N
        cf_recs_sorted = cf_recs.sort_values("cf_score_raw", ascending=False).head(TOP_N)
        for _, row in cf_recs_sorted.iterrows():
            appid = int(row["appid"])
            title = appid_to_title.get(appid, f"appid {appid}")
            cf_val = row.get("cf_score_norm", np.nan)
            try:
                cf_str = f"{float(cf_val):.4f}" if np.isfinite(cf_val) else "nan"
            except Exception:
                cf_str = "nan"

            print(
                f"- {title} "
                f"(appid={appid}, cf_norm={cf_str})"
            )
        return
    # ---------- END pure CF fallback branch ----------

    cbf_recs = cbf_recs.copy()

    # Base CBF signal
    if "cbf_anchor_combined" in cbf_recs.columns:
        cbf_recs["cbf_score_raw"] = cbf_recs["cbf_anchor_combined"].astype(float)
    elif "cbf" in cbf_recs.columns:
        cbf_recs["cbf_score_raw"] = cbf_recs["cbf"].astype(float)
    else:
        # fallback: use a neutral constant (should not really happen)
        cbf_recs["cbf_score_raw"] = 0.0

    cbf_candidates = cbf_recs[["appid", "title", "cbf_score_raw"]].copy()

    # --------------------------------------------------
    # 2) CF: get its own top-K candidates
    cf_recs = generate_cf_recommendations(
        steamid64=steamid64,
        top_k=CANDIDATE_POOL_SIZE,
    )

    # --------------------------------------------------
    # 3) Union of CBF + CF candidates
    if cf_recs is not None and not cf_recs.empty:
        union = pd.merge(
            cbf_candidates,
            cf_recs[["appid", "cf_score_raw"]],
            on="appid",
            how="outer",
        )
    else:
        # CF unavailable → union is just CBF with cf_score_raw = 0
        union = cbf_candidates.copy()
        union["cf_score_raw"] = 0.0

    union["cbf_score_raw"] = union["cbf_score_raw"].fillna(0.0)
    union["cf_score_raw"] = union["cf_score_raw"].fillna(0.0)

    # --------------------------------------------------
    # 4) Normalise CBF and CF scores separately, then hybrid
    union["cbf_score_norm"] = normalise_scores(
        union["cbf_score_raw"].to_numpy(dtype=float)
    )
    union["cf_score_norm"] = normalise_scores(
        union["cf_score_raw"].to_numpy(dtype=float)
    )

    cbf_norm_arr = union["cbf_score_norm"].to_numpy(dtype=float)
    cf_norm_arr = union["cf_score_norm"].to_numpy(dtype=float)

    if np.all(cf_norm_arr == 0.0):
        print("\n[HYBRID] CF unavailable or zero scores; using pure CBF for hybrid.")
        hybrid_arr = cbf_norm_arr.copy()
    else:
        hybrid_arr = combine_cbf_cf(
            cbf_scores=cbf_norm_arr,
            cf_scores=cf_norm_arr,
            alpha=ALPHA_HYBRID,
        )

    union["hybrid_score"] = hybrid_arr

    # --------------------------------------------------
    # 5) Single MMR pass on HYBRID scores
    appids_union = union["appid"].astype(int).to_numpy()
    selected_indices = mmr_rerank(
        appids=appids_union,
        hybrid_scores=hybrid_arr,
        lambda_mmr=LAMBDA_MMR_HYBRID,
        top_k=TOP_N,
    )

    final = union.iloc[selected_indices].reset_index(drop=True)

    # --------------------------------------------------
    # 6) Display results (showing normalised scores)
    print("\nTop Recommendations (Hybrid CBF + CF with MMR on hybrid):")
    for _, row in final.iterrows():
        title = row.get("title")
        if not isinstance(title, str) or not title.strip():
            title = f"appid {row['appid']}"

        cbf_val = row.get("cbf_score_norm", np.nan)
        cf_val = row.get("cf_score_norm", np.nan)
        hybrid_val = row.get("hybrid_score", np.nan)

        try:
            cbf_str = f"{float(cbf_val):.4f}" if np.isfinite(cbf_val) else "nan"
        except Exception:
            cbf_str = "nan"

        try:
            cf_str = f"{float(cf_val):.4f}" if np.isfinite(cf_val) else "nan"
        except Exception:
            cf_str = "nan"

        try:
            hybrid_str = f"{float(hybrid_val):.4f}" if np.isfinite(hybrid_val) else "nan"
        except Exception:
            hybrid_str = "nan"

        print(
            f"- {title} "
            f"(appid={row['appid']}, "
            f"cbf_norm={cbf_str}, "
            f"cf_norm={cf_str}, "
            f"hybrid={hybrid_str})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ludex hybrid recommender (CBF + CF + single MMR on hybrid)."
    )
    parser.add_argument(
        "steamid64",
        help="SteamID64 of the user to recommend games for.",
    )
    args = parser.parse_args()

    main(args.steamid64)