"""
CF_recommend
------------

Collaborative-filtering (ALS) recommendation pipeline for a single Steam user.

This module encapsulates the CF-only orchestration (simplified from the old
`recommend_for_user.py`) and exposes a DataFrame-based API:

    from CF.CF_recommend import generate_cf_recommendations

    cf_recs = generate_cf_recommendations(
        steamid64=...,
        top_k=1000,
    )

Returns
-------
    cf_recs : pandas.DataFrame
        Columns:
            - appid    : int
            - cf_score_raw : float (popularity-adjusted ALS score)
"""

from __future__ import annotations

from typing import List, Dict, Set

import numpy as np
import pandas as pd

from .cf_model import load_cf_model, INTERACTIONS_CSV
from .interactions_update import (
    load_interactions,
    ensure_users_in_data_and_retrain,
)


def _build_popularity(df: pd.DataFrame) -> Dict[int, float]:
    """
    From interactions, build normalised popularity score per appid.
    """
    pop = df.groupby("appid")["steamid"].nunique().astype(float)
    pop_min, pop_max = pop.min(), pop.max()
    if pop_max > pop_min:
        pop_norm = (pop - pop_min) / (pop_max - pop_min)
    elif pop_max > 0:
        pop_norm = pop / pop_max
    else:
        pop_norm = pop
    return pop_norm.to_dict()


def generate_cf_recommendations(
    steamid64: str,
    *,
    top_k: int = 1000,
) -> pd.DataFrame:
    """
    Run the CF (ALS) pipeline for a single user and return a CF-only
    recommendation DataFrame.

    Parameters
    ----------
    steamid64 : str
        SteamID64 of the user.
    top_k : int, default 1000
        Number of CF candidates to return.

    Returns
    -------
    pandas.DataFrame
        Columns:
            - appid    : int
            - cf_score_raw : float (pop-adjusted ALS score)

        May be empty if CF is unavailable or user cannot be scored.
    """
    steamid64 = str(steamid64).strip()
    if not steamid64:
        raise ValueError("steamid64 must be a non-empty string")

    # --------------------------------------------------
    # 1) Ensure interactions CSV exists
    if not INTERACTIONS_CSV.exists():
        # CF pipeline not ready → no CF candidates
        return pd.DataFrame(columns=["appid", "cf_score_raw"])

    # --------------------------------------------------
    # 2) Enrich interactions with this user (auto retrain if needed)
    try:
        ensure_users_in_data_and_retrain([steamid64])
    except Exception as e:
        print(f"[CF] Warning: ensure_users_in_data_and_retrain failed: {e!r}")
        return pd.DataFrame(columns=["appid", "cf_score_raw"])

    # --------------------------------------------------
    # 3) Load interactions + popularity
    df = load_interactions()
    if df.empty:
        return pd.DataFrame(columns=["appid", "cf_score_raw"])

    pop_norm = _build_popularity(df)

    # Owned games in training (to filter them out)
    user_inter = df[df["steamid"].astype(str) == steamid64]
    owned_appids_training: Set[int] = set(user_inter["appid"].astype(int).tolist())

    # --------------------------------------------------
    # 4) Load CF ALS model + IDs
    try:
        model, user_ids, item_ids = load_cf_model(force_retrain=False)
    except Exception as e:
        print(f"[CF] Warning: load_cf_model failed: {e!r}")
        return pd.DataFrame(columns=["appid", "cf_score_raw"])

    user_ids = [str(sid) for sid in user_ids]
    item_ids = list(item_ids)

    if steamid64 not in user_ids:
        print(f"[CF] User {steamid64} not in ALS model after enrichment; CF empty.")
        return pd.DataFrame(columns=["appid", "cf_score_raw"])

    user_idx = user_ids.index(steamid64)

    # --------------------------------------------------
    # 5) Recommend via ALS
    total_items = len(item_ids)
    raw_N = min(top_k + len(owned_appids_training) + 200, total_items)

    try:
        item_indices, scores = model.recommend(
            userid=user_idx,
            user_items=None,
            N=raw_N,
            filter_already_liked_items=False,
        )
    except Exception as e:
        print(f"[CF] Warning: ALS recommend failed: {e!r}")
        return pd.DataFrame(columns=["appid", "cf_score_raw"])

    # --------------------------------------------------
    # 6) Popularity-adjusted CF score and filter owned
    LAMBDA_POP = 0.5  # popularity penalty factor

    recs: List[tuple[int, float]] = []
    for item_idx, s in zip(item_indices, scores):
        if item_idx < 0 or item_idx >= len(item_ids):
            continue
        appid = int(item_ids[item_idx])

        # Skip games already in training interactions for this user
        if appid in owned_appids_training:
            continue

        base_score = float(s)
        p = pop_norm.get(appid, 0.0)
        adj_score = base_score * (1.0 - LAMBDA_POP * p)
        recs.append((appid, adj_score))

    if not recs:
        return pd.DataFrame(columns=["appid", "cf_score_raw"])

    # Sort and keep top_k
    recs.sort(key=lambda x: x[1], reverse=True)
    recs = recs[:top_k]

    df_out = pd.DataFrame(recs, columns=["appid", "cf_score_raw"])
    return df_out