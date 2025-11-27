import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
from typing import Optional, Sequence, List, Tuple

from sklearn.preprocessing import normalize


# ======================================================
# STEAM FETCH + MAPPING

def fetch_owned_games(steamid64: str, api_key: str) -> pd.DataFrame:
    """
    Call the Steam Web API to fetch owned games + playtime.

    Returns a DataFrame with columns:
        ['appid', 'title', 'playtime_min']
    """
    url = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
    params = {
        "key": api_key,
        "steamid": steamid64,
        "include_appinfo": 1,
        "include_played_free_games": 1,
        "format": "json",
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()

    games = resp.json().get("response", {}).get("games", [])
    if not games:
        print("No games returned for this user.")
        return pd.DataFrame(columns=["appid", "title", "playtime_min"])

    df = pd.DataFrame(games)
    df = df.rename(
        columns={
            "appid": "appid",
            "playtime_forever": "playtime_min",
            "name": "title",
        }
    )
    df = df[["appid", "title", "playtime_min"]]
    df = df.sort_values("playtime_min", ascending=False)
    return df


def map_owned_to_indices(owned_df: pd.DataFrame, catalogue_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map owned games (by appid) to row indices in the catalogue DataFrame.

    Adds a 'row_idx' column to owned_df (dropping any games not in the catalogue).
    """
    appid_to_idx = {int(a): i for i, a in enumerate(catalogue_df["appid"])}

    owned_df = owned_df.copy()
    owned_df["row_idx"] = owned_df["appid"].map(appid_to_idx)

    owned_df = owned_df.dropna(subset=["row_idx"])
    owned_df["row_idx"] = owned_df["row_idx"].astype(int)
    return owned_df


# ======================================================
# INTERNAL: ANCHOR SELECTION

def _select_anchors(
    owned_mapped: pd.DataFrame,
    min_playtime: int = 60,
) -> Optional[pd.DataFrame]:
    """
    Internal helper to select anchor games from the user's library.

    Logic:
      - Prefer ALL games with playtime >= min_playtime.
      - If there are none, fall back to ALL games with playtime > 0.
      - If user has no games with playtime > 0, return None.

    Returns:
      anchors_df: DataFrame with at least ['row_idx', 'playtime_min'] or None.
    """
    owned_mapped = owned_mapped.copy()

    strict_anchors = owned_mapped[owned_mapped["playtime_min"] >= min_playtime].copy()
    if not strict_anchors.empty:
        anchors_df = strict_anchors.sort_values("playtime_min", ascending=False)
        print(
            f"Using {len(anchors_df)} games with playtime >= {min_playtime} minutes "
            "as anchors."
        )
    else:
        print(
            f"User has no games with playtime >= {min_playtime} minutes. "
            "Falling back to all games with non-zero playtime (cold-start contingency)."
        )
        fallback_candidates = owned_mapped[owned_mapped["playtime_min"] > 0].copy()
        if fallback_candidates.empty:
            print(
                "User has no games with non-zero playtime. "
                "Cannot select anchors (pure cold-start)."
            )
            return None

        anchors_df = fallback_candidates.sort_values("playtime_min", ascending=False)
        print(
            f"Using {len(anchors_df)} games with playtime > 0 minutes "
            "as anchors."
        )

    if anchors_df.empty:
        return None

    return anchors_df


# ======================================================
# USER CONTENT PROFILE (SINGLE VECTOR, WITH COLD-START HANDLING)

def build_user_content_profile(
    owned_mapped: pd.DataFrame,
    full_matrix_norm: sp.csr_matrix,
    min_playtime: int = 60,
    min_games_for_strict: int = 5,  # kept for backward compatibility (unused)
    fallback_top_k: int = 5,        # kept for backward compatibility (unused)
    max_strict_anchors: int = 10,   # kept for backward compatibility (unused)
) -> Optional[np.ndarray]:
    """
    Build a single user content profile vector v_u in the same space as the games.

    Updated logic (simple and robust):
      - Use *all* games with playtime >= min_playtime as anchors.
      - If there are none, fall back to *all* games with playtime > 0.
      - If the user has no games with playtime > 0, return None (true cold-start).

    Steps (once anchors are chosen):
        1) Turn playtime into log-scaled weights w_i = log(1 + p_i).
        2) Normalise weights to sum to 1.
        3) Compute weighted average of their L2-normalised feature vectors.
        4) L2-normalise the resulting user vector.

    Returns:
        user_vec: np.ndarray of shape (d,) with ||user_vec||_2 = 1,
                  or None if we cannot build a profile.
    """
    anchors_df = _select_anchors(owned_mapped, min_playtime=min_playtime)
    if anchors_df is None or anchors_df.empty:
        print("No anchors available to build user profile.")
        return None

    row_idxs = anchors_df["row_idx"].to_numpy()
    playtimes = anchors_df["playtime_min"].to_numpy(dtype=float)

    # ----- log-scaled playtime weights, normalised to sum=1 -----
    w_tilde = np.log1p(playtimes)  # log(1 + p)
    weight_sum = w_tilde.sum()
    if weight_sum <= 0:
        print("Non-positive weight sum – cannot build user profile.")
        return None

    weights = w_tilde / weight_sum  # shape (k,)

    # ----- Weighted average of game vectors -----
    subset = full_matrix_norm[row_idxs]            # (k, d) sparse CSR
    weighted = subset.multiply(weights.reshape(-1, 1))
    user_vec = weighted.sum(axis=0)               # (1, d)

    user_vec = np.asarray(user_vec).ravel()       # (d,)

    # ----- L2-normalise the user vector -----
    norm = np.linalg.norm(user_vec)
    if norm == 0.0:
        print("User vector has zero norm – cannot normalise.")
        return None

    user_vec = user_vec / norm
    return user_vec


# ======================================================
# GLOBAL CBF SCORING

def score_games_cbf(
    user_vec: np.ndarray,
    full_matrix_norm: sp.csr_matrix,
) -> np.ndarray:
    """
    Given a user content profile vector v_u and the game feature matrix F,
    compute CBF scores for ALL games:

        CBF(u, i) = v_u · f_i  (cosine similarity)

    Assumes:
        - full_matrix_norm: CSR (n_games, d), rows L2-normalised.
        - user_vec: np.ndarray (d,), L2-normalised.

    Returns:
        scores: np.ndarray of shape (n_games,), where scores[i] = CBF(u, i).
    """
    if user_vec.ndim != 1:
        user_vec = user_vec.ravel()

    scores = full_matrix_norm.dot(user_vec)  # (n, d) ⋅ (d,) → (n,)
    return np.asarray(scores).ravel()


# ======================================================
# ANCHOR-BASED SCORES (SOFT, WEIGHTED BY PLAYTIME)

def compute_anchor_soft_scores(
    owned_mapped: pd.DataFrame,
    full_matrix_norm: sp.csr_matrix,
    candidate_indices: np.ndarray,
    min_playtime: int = 60,
) -> np.ndarray:
    """
    Compute a soft anchor-based similarity score for each candidate game.

    For each anchor 'a':
        - anchor embedding is f_a (row in full_matrix_norm)
        - raw weight w̃_a = log(1 + playtime_a)
        - normalised weight w_a = w̃_a / Σ_b w̃_b

    For each candidate i in candidate_indices:
        anchor_soft[i] = Σ_a w_a * (f_i ⋅ f_a)

    Returns:
        anchor_soft: np.ndarray of shape (len(candidate_indices),)
    """
    anchors_df = _select_anchors(owned_mapped, min_playtime=min_playtime)
    if anchors_df is None or anchors_df.empty:
        # No anchors → no anchor influence
        return np.zeros(len(candidate_indices), dtype=float)

    anchor_row_idxs = anchors_df["row_idx"].to_numpy()
    playtimes = anchors_df["playtime_min"].to_numpy(dtype=float)

    w_tilde = np.log1p(playtimes)
    weight_sum = w_tilde.sum()
    if weight_sum <= 0:
        return np.zeros(len(candidate_indices), dtype=float)

    weights = w_tilde / weight_sum  # shape (k,)

    # Candidate and anchor matrices
    cand_mat = full_matrix_norm[candidate_indices]    # (m, d)
    anchor_mat = full_matrix_norm[anchor_row_idxs]    # (k, d)

    # Similarity matrix: (m, d) * (d, k) → (m, k)
    sims = cand_mat.dot(anchor_mat.T).toarray()       # dense is fine at this scale

    # Weighted sum over anchors → (m,)
    anchor_soft = sims.dot(weights)
    return anchor_soft


# ======================================================
# DIVERSITY-AWARE RE-RANKING (MMR)

def mmr_rerank(
    candidate_indices: np.ndarray,
    combined_scores: np.ndarray,
    full_matrix_norm: sp.csr_matrix,
    top_n: int = 20,
    lambda_mmr: float = 0.7,
) -> List[int]:
    """
    Maximal Marginal Relevance (MMR) re-ranking to trade off
    relevance (combined_scores) vs diversity (cosine similarity
    to already-selected items).

    For each step:
        score_mmr(i) = λ * combined[i] - (1 - λ) * max_{j in S} sim(i, j)

    Where:
        - sim(i, j) = f_i ⋅ f_j  (cosine, rows of full_matrix_norm)
        - S is the set of already selected items (by global index)

    Returns:
        ordered_selected: list of global row indices (subset of candidate_indices)
    """
    if len(candidate_indices) == 0:
        return []

    top_n = min(top_n, len(candidate_indices))
    combined_scores = np.asarray(combined_scores).ravel()

    remaining = list(range(len(candidate_indices)))  # indices into candidate_indices
    selected_global: List[int] = []

    for _ in range(top_n):
        best_local = None
        best_score = -np.inf

        for local_idx in remaining:
            global_idx = candidate_indices[local_idx]
            relevance = combined_scores[local_idx]

            if not selected_global:
                mmr_score = relevance
            else:
                # Compute max similarity to already selected items
                sims = full_matrix_norm[global_idx].dot(
                    full_matrix_norm[selected_global].T
                ).toarray().ravel()
                max_sim = float(sims.max()) if sims.size > 0 else 0.0

                mmr_score = lambda_mmr * relevance - (1.0 - lambda_mmr) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_local = local_idx

        if best_local is None:
            break

        # Add the best item (by global index) to the selection
        best_global = int(candidate_indices[best_local])
        selected_global.append(best_global)
        remaining.remove(best_local)

    return selected_global


# ======================================================
# HIGH-LEVEL RECOMMENDER: USER VECTOR + ANCHORS + MMR

def recommend_cbf_user_plus_anchors_mmr(
    catalogue_df: pd.DataFrame,
    full_matrix_norm: sp.csr_matrix,
    owned_mapped: pd.DataFrame,
    user_vec: np.ndarray,
    top_n: int = 20,
    candidate_pool_size: int = 500,
    min_playtime: int = 60,
    beta: float = 0.3,
    lambda_mmr: float = 0.7,
) -> pd.DataFrame:
    """
    Main CBF recommendation routine combining:

      1) Global user-vector relevance
      2) Anchor-based similarity (soft, weighted by playtime)
      3) Diversity-aware MMR re-ranking

    Steps:
      - Compute global CBF scores for all games.
      - Build candidate pool C with top `candidate_pool_size` non-owned games.
      - Compute anchor_soft[i] over C.
      - Blend:
            combined_raw[i] = (1 - beta) * global_score[i] + beta * anchor_soft[i]
      - Re-rank C with MMR:
            score_mmr(i) = λ * combined_raw[i] - (1 - λ) * max_{j∈S} sim(i, j)
      - Return top-N as a DataFrame slice of catalogue_df, with extra score columns.

    Returns:
      recs_df: DataFrame with at least ['appid', 'title', 'cbf', 'cbf_anchor_combined'].
    """
    if user_vec is None:
        raise ValueError("user_vec is None. Build user content profile first.")

    # 1) Global CBF scores for all games
    global_scores = score_games_cbf(user_vec, full_matrix_norm)

    df_scores = catalogue_df.copy()
    df_scores["cbf"] = global_scores

    owned_appids = set(owned_mapped["appid"])
    df_candidates = df_scores[~df_scores["appid"].isin(owned_appids)].copy()

    # 2) Candidate pool by top global CBF
    df_candidates = df_candidates.sort_values("cbf", ascending=False)
    if candidate_pool_size is not None and candidate_pool_size > 0:
        df_candidates = df_candidates.head(candidate_pool_size)

    if df_candidates.empty:
        print("No candidate games available for recommendation.")
        return pd.DataFrame()

    candidate_indices = df_candidates.index.to_numpy()

    # 3) Anchor-based soft scores on candidate pool
    anchor_soft = compute_anchor_soft_scores(
        owned_mapped=owned_mapped,
        full_matrix_norm=full_matrix_norm,
        candidate_indices=candidate_indices,
        min_playtime=min_playtime,
    )

    # 4) Blend global user-vector score + anchor soft score
    combined_raw = (1.0 - beta) * df_candidates["cbf"].to_numpy() + beta * anchor_soft
    df_candidates["cbf_anchor_combined"] = combined_raw

    # 5) Diversity-aware re-ranking with MMR
    selected_global_indices = mmr_rerank(
        candidate_indices=candidate_indices,
        combined_scores=combined_raw,
        full_matrix_norm=full_matrix_norm,
        top_n=top_n,
        lambda_mmr=lambda_mmr,
    )

    if not selected_global_indices:
        print("MMR re-ranking returned no items.")
        return pd.DataFrame()

    # Build final recommendations DataFrame in the chosen order
    ordered_idx = selected_global_indices
    recs_df = df_scores.loc[ordered_idx].copy()

    # Attach final scores (cbf + combined)
    # We map from candidate table (which has the combined scores) where possible.
    combined_map = df_candidates["cbf_anchor_combined"].to_dict()
    cbf_map = df_candidates["cbf"].to_dict()

    recs_df["cbf"] = recs_df.index.map(cbf_map)
    recs_df["cbf_anchor_combined"] = recs_df.index.map(combined_map)

    return recs_df
