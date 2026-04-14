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

CATALOGUE_DF = None
FULL_MATRIX = None

def get_cbf_data():
    global CATALOGUE_DF, FULL_MATRIX
    if CATALOGUE_DF is None or FULL_MATRIX is None:
        from CBF.cbf_model import load_catalogue_and_features
        CATALOGUE_DF, FULL_MATRIX = load_catalogue_and_features()
    return CATALOGUE_DF, FULL_MATRIX

# --- HYBRID SETTINGS ---
TOP_N = 20
CANDIDATE_POOL_SIZE = 50  
MIN_PLAYTIME = 60
LAMBDA_MMR_CBF = 0.0
LAMBDA_MMR_HYBRID = 0.6
BETA_ANCHOR_BLEND = 0.3
ALPHA_HYBRID = 0.35
METADATA_THREADS = 2    


# ======================================================
# FLASK SETUP
# ======================================================
app = Flask(__name__)
app.secret_key = os.environ.get(
    "LUDEX_SECRET",
    "957e9cc4737b9aa0841fbd16e87ff4b30a9d72fa8d0cf0d246424c5054199cc4"
)

STEAM_API_KEY = os.environ.get("STEAM_API_KEY", "B0628D7BA865C799E6D9679396DC563B")
APP_URL = os.environ.get("APP_URL")

if not APP_URL:
    APP_URL = "http://127.0.0.1:5000"

realm = APP_URL
return_to = f"{APP_URL}/authorize"

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
# Now using get_cbf_data()


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
    catalogue_df, full_matrix_norm = get_cbf_data()
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
    sim_matrix = cand_vecs @ cand_vecs.T

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

    from CF.CF_recommend import generate_cf_recommendations
    from CBF.CBF_recommend import generate_cbf_recommendations
    import numpy as np
    import pandas as pd

    catalogue_df, full_matrix = get_cbf_data()

    cbf_recs = generate_cbf_recommendations(
        steamid64=str(steamid),
        api_key=STEAM_API_KEY,
        top_n=25
    )

    cf_recs = generate_cf_recommendations(
        steamid64=str(steamid),
        top_k=25
    )

    if cbf_recs is None or cbf_recs.empty:
        merged = cf_recs.copy()
    elif cf_recs is None or cf_recs.empty:
        merged = cbf_recs.copy()
    else:
        merged = pd.merge(cbf_recs, cf_recs, on="appid", how="outer")

    # Safely extract CBF score (using the anchor-blended score for better accuracy)
    merged["cbf_score"] = merged["cbf_anchor_combined"].fillna(0) if "cbf_anchor_combined" in merged.columns else 0
    
    # Safely extract CF score
    merged["cf_score"] = merged["cf_score_raw"].fillna(0) if "cf_score_raw" in merged.columns else 0

    merged["score"] = 0.6 * merged["cbf_score"] + 0.4 * merged["cf_score"]

    candidates = merged.sort_values("score", ascending=False).head(25).reset_index(drop=True)

    appid_to_idx = {int(a): i for i, a in enumerate(catalogue_df["appid"])}

    valid_rows = []
    valid_indices = []
    for i, appid in enumerate(candidates["appid"]):
        idx = appid_to_idx.get(int(appid), -1)
        if idx >= 0:
            valid_rows.append(idx)
            valid_indices.append(i)
    valid_rows = valid_rows[:20]
    valid_indices = valid_indices[:20]

    if not valid_rows:
        final = candidates.head(20)
    else:
        scores = candidates.iloc[valid_indices]["score"].values
        selected = []
        selected_global = []

        for _ in range(min(20, len(valid_rows))):
            if not selected:
                idx = int(np.argmax(scores))
            else:
                sims = []
                for i, global_idx in enumerate(valid_rows):
                    if i in selected:
                        sims.append(0)
                        continue

                    row_vec = full_matrix[global_idx]
                    sel_vecs = full_matrix[[valid_rows[j] for j in selected]]

                    sim = row_vec.dot(sel_vecs.T)
                    sim = sim.toarray().ravel() # <--- THE FIX
                    sims.append(sim.max() if sim.size > 0 else 0)

                max_sim = np.array(sims)
                mmr = 0.7 * scores - 0.3 * max_sim
                mmr[selected] = -np.inf
                idx = int(np.argmax(mmr))

            selected.append(idx)

        final = candidates.iloc[[valid_indices[i] for i in selected]]
    recs = final.to_dict(orient="records")

    from concurrent.futures import ThreadPoolExecutor

    def enrich(r):
        meta = cached_store_metadata(str(r["appid"]))
        r.update(meta)
        return r

    with ThreadPoolExecutor(max_workers=METADATA_THREADS) as exe:
        futures = [exe.submit(enrich, r) for r in recs]
        recs = [f.result() for f in futures]

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
            get_cbf_data()
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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
