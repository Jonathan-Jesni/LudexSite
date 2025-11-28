from flask import Flask, redirect, request, session, render_template, url_for
from steam_openid import SteamOpenID
import requests
import os
import time
import numpy as np
import pandas as pd
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- HYBRID MODULES ---
from CBF.CBF_recommend import generate_cbf_recommendations
from CBF.cbf_model import load_catalogue_and_features
from CF.CF_recommend import generate_cf_recommendations

# --- HYBRID SETTINGS ---
TOP_N = 20
CANDIDATE_POOL_SIZE = 300  
MIN_PLAYTIME = 60
LAMBDA_MMR_CBF = 0.0
LAMBDA_MMR_HYBRID = 0.6
BETA_ANCHOR_BLEND = 0.3
ALPHA_HYBRID = 0.35
METADATA_THREADS = 4    


# ======================================================
# FLASK SETUP
# ======================================================
app = Flask(__name__)
app.secret_key = os.environ.get(
    "LUDEX_SECRET",
    "957e9cc4737b9aa0841fbd16e87ff4b30a9d72fa8d0cf0d246424c5054199cc4"
)

STEAM_API_KEY = os.environ.get("STEAM_API_KEY", "B0628D7BA865C799E6D9679396DC563B")
realm = "http://127.0.0.1:5000/"
return_to = "http://127.0.0.1:5000/authorize"


# ======================================================
# METADATA FETCH
# ======================================================
def fetch_store_metadata(appid: str) -> dict:
    try:
        r = requests.get(
            "https://store.steampowered.com/api/appdetails",
            params={"appids": appid, "cc": "us", "l": "english"},
            timeout=4
        )
        data = r.json().get(str(appid), {}).get("data")
        if not data:
            raise Exception()

        price_overview = data.get("price_overview", {})
        price = (
            price_overview.get("final_formatted")
            or ("Free" if data.get("is_free") else "N/A")
        )

        return {
            "name": data.get("name", ""),
            "image": data.get("header_image", ""),
            "short_description": data.get("short_description", ""),
            "genres": [g["description"] for g in data.get("genres", [])],
            "price": price,
            "metacritic": data.get("metacritic", {}).get("score"),
            "release_date": data.get("release_date", {}).get("date", ""),
            "recommendations": data.get("recommendations", {}).get("total"),
        }

    except Exception:
        return {
            "name": "",
            "image": "",
            "short_description": "",
            "genres": [],
            "price": "N/A",
            "metacritic": None,
            "release_date": "",
            "recommendations": None,
        }


@lru_cache(maxsize=512)
def cached_store_metadata(appid: str):
    return fetch_store_metadata(appid)


# ======================================================
# LOAD CATALOGUE + MATRIX WITH CACHE
# ======================================================
@lru_cache(maxsize=1)
def load_cached_catalogue_and_features():
    return load_catalogue_and_features()


# ======================================================
# SIMPLE NORMALISATION
# ======================================================
def normalise_scores(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    lo, hi = arr.min(), arr.max()
    if hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ======================================================
# MMR
# ======================================================
def mmr_rerank(appids, hybrid_scores, lambda_mmr, top_k):
    catalogue_df, full_matrix_norm = load_cached_catalogue_and_features()
    appid_to_idx = {int(a): i for i, a in enumerate(catalogue_df["appid"].astype(int))}

    appids = np.asarray(appids, dtype=int)
    hybrid_scores = np.asarray(hybrid_scores, dtype=float)

    row_idx = np.array([appid_to_idx.get(int(a), -1) for a in appids])
    mask = row_idx >= 0

    if not np.any(mask):
        return np.argsort(-hybrid_scores)[:top_k]

    valid_positions = np.where(mask)[0]
    mat_rows = row_idx[mask]

    cand_vecs = full_matrix_norm[mat_rows]
    sim_matrix = (cand_vecs @ cand_vecs.T).toarray()

    selected_local = []
    r_valid = hybrid_scores[mask]
    max_items = min(top_k, len(valid_positions))

    while len(selected_local) < max_items:
        if not selected_local:
            best = int(np.argmax(r_valid))
        else:
            sel = np.array(selected_local)
            max_sims = sim_matrix[:, sel].max(axis=1)
            mmr = lambda_mmr * r_valid - (1 - lambda_mmr) * max_sims
            mmr[sel] = -np.inf
            best = int(np.argmax(mmr))
            if np.isneginf(mmr[best]):
                break
        selected_local.append(best)

    selected_global = [valid_positions[i] for i in selected_local]

    if len(selected_global) < top_k:
        remaining = [i for i in range(len(appids)) if i not in selected_global]
        remaining_sorted = sorted(remaining, key=lambda i: hybrid_scores[i], reverse=True)
        need = top_k - len(selected_global)
        selected_global.extend(remaining_sorted[:need])

    return np.array(selected_global, dtype=int)


# ======================================================
# ROUTES
# ======================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login")
def login():
    return redirect(SteamOpenID(realm=realm, return_to=return_to).get_redirect_url())


@app.route("/authorize")
def authorize():
    steam = SteamOpenID(realm=realm, return_to=return_to)
    steamid = steam.validate_results(request.args)
    if not steamid:
        return "Steam authentication failed"

    r = requests.get(
        "https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v2/",
        params={"key": STEAM_API_KEY, "steamids": steamid}
    )
    player = r.json().get("response", {}).get("players", [{}])[0]

    session.update({
        "steamid": steamid,
        "username": player.get("personaname", "Unknown"),
        "avatar": player.get("avatarfull", "/static/default-avatar.png"),
        "country": player.get("loccountrycode", "US").lower(),
    })

    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    if "steamid" not in session:
        return redirect(url_for("index"))
    return render_template("dashboard.html", **session)


# ======================================================
# HYBRID RECOMMENDER
# ======================================================
@app.route("/recommend")
def recommend():
    import time
    t0 = time.time()

    print("\n===== HYBRID TIMING DEBUG =====")

    if "steamid" not in session:
        return redirect(url_for("index"))
    steamid = session["steamid"]

    print("0) start:", time.time() - t0)

    # 1) CBF
    cbf = generate_cbf_recommendations(
        steamid64=steamid,
        api_key=STEAM_API_KEY,
        top_n=CANDIDATE_POOL_SIZE,
        min_playtime=MIN_PLAYTIME,
        candidate_pool_size=CANDIDATE_POOL_SIZE,
        beta_anchor_blend=BETA_ANCHOR_BLEND,
        lambda_mmr=LAMBDA_MMR_CBF
    )
    print("1) CBF done:", time.time() - t0)

    if "cbf_anchor_combined" in cbf.columns:
        cbf["cbf_score_raw"] = cbf["cbf_anchor_combined"]
    else:
        cbf["cbf_score_raw"] = 0.0
    cbf_df = cbf[["appid", "title", "cbf_score_raw"]]

    # 2) CF
    cf = generate_cf_recommendations(
        steamid64=steamid,
        top_k=CANDIDATE_POOL_SIZE
    )
    print("2) CF done:", time.time() - t0)

    # 3) Merge
    if cf is not None and not cf.empty:
        union = pd.merge(
            cbf_df,
            cf[["appid", "cf_score_raw"]],
            on="appid",
            how="outer"
        )
    else:
        union = cbf_df.copy()
        union["cf_score_raw"] = 0.0

    print("3) merge:", time.time() - t0)

    # Normalize
    union = union.fillna({"cbf_score_raw": 0.0, "cf_score_raw": 0.0})
    union["cbf_norm"] = normalise_scores(union["cbf_score_raw"])
    union["cf_norm"] = normalise_scores(union["cf_score_raw"])
    union["hybrid"] = (
        ALPHA_HYBRID * union["cf_norm"] +
        (1 - ALPHA_HYBRID) * union["cbf_norm"]
    )

    print("4) hybrid computed:", time.time() - t0)

    # 4) MMR
    selected = mmr_rerank(
        appids=union["appid"].astype(int).to_numpy(),
        hybrid_scores=union["hybrid"].to_numpy(),
        lambda_mmr=LAMBDA_MMR_HYBRID,
        top_k=TOP_N
    )
    print("5) MMR rerank:", time.time() - t0)

    # Convert to list
    final = union.iloc[selected].reset_index(drop=True)
    recs = final.to_dict(orient="records")

    # 5) Metadata
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def enrich(r):
        meta = cached_store_metadata(str(r["appid"]))
        r.update(meta)
        return r

    with ThreadPoolExecutor(max_workers=METADATA_THREADS) as exe:
        futures = [exe.submit(enrich, r) for r in recs]
        recs = [f.result() for f in as_completed(futures)]

    print("6) metadata:", time.time() - t0)
    print("===== END TIMING =====\n")

    return render_template("recommend.html", recs=recs, **session)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# ======================================================
# CACHE PREWARM (Flask 3.x SAFE)
# ======================================================
def warm_cache_async():
    app.logger.info("Pre-warming catalogue/features cache in background...")

    def _load():
        try:
            load_cached_catalogue_and_features()
            app.logger.info("Pre-warm complete.")
        except Exception:
            app.logger.exception("Pre-warm failed.")

    from threading import Thread
    Thread(target=_load, daemon=True).start()


# run prewarm immediately at startup
warm_cache_async()


# ======================================================
# RUN APP
# ======================================================
if __name__ == "__main__":
    app.run(debug=True)
