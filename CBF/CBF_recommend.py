"""
CBF_recommend
-------------

Content-based (TF–IDF) recommendation pipeline for a single Steam user.

This module contains the *CBF-only* orchestration that used to live in
main.py, mirroring how CF_recommend.py encapsulates CF logic.

Public API
----------
    from CBF.CBF_recommend import generate_cbf_recommendations

    recs = generate_cbf_recommendations(
        steamid64=...,
        api_key=...,
        top_n=20,
        min_playtime=60,
        candidate_pool_size=500,
        beta_anchor_blend=0.3,
        lambda_mmr=0.7,
    )

Returns
-------
    recs : pandas.DataFrame
        Recommendation table returned by
        `recommend_cbf_user_plus_anchors_mmr` (with columns like
        appid, title, cbf, cbf_anchor_combined, etc.).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .cbf_model import load_catalogue_and_features
from .user import (
    fetch_owned_games,
    map_owned_to_indices,
    build_user_content_profile,
    recommend_cbf_user_plus_anchors_mmr,
)
from .catalogue_update import ensure_user_games_in_catalogue_and_refresh


def generate_cbf_recommendations(
    steamid64: str,
    api_key: str,
    *,
    top_n: int = 20,
    min_playtime: int = 60,
    candidate_pool_size: int = 500,
    beta_anchor_blend: float = 0.3,
    lambda_mmr: float = 0.7,
) -> pd.DataFrame:
    """
    Run the full CBF pipeline for a single user and return a
    recommendation DataFrame.

    This wraps the previous main.py steps:
      1) Load catalogue + feature matrix
      2) Fetch owned games
      3) Ensure owned games exist in catalogue (extend + rebuild if needed)
      4) Map owned games into catalogue indices
      5) Build a single user content profile vector
      6) Score all games using user vector + anchors + MMR

    Parameters
    ----------
    steamid64 : str
        SteamID64 of the user.
    api_key : str
        Steam Web API key (already loaded by caller).
    top_n : int, default 20
        Number of final recommendations to return.
    min_playtime : int, default 60
        Minimum minutes for an owned game to count strongly.
    candidate_pool_size : int, default 500
        How many top CBF games to consider before MMR.
    beta_anchor_blend : float, default 0.3
        Blend weight for anchor_soft vs global CBF in the user scoring.
    lambda_mmr : float, default 0.7
        Relevance vs diversity trade-off for MMR re-ranking.

    Returns
    -------
    pandas.DataFrame
        Recommendation DataFrame produced by
        `recommend_cbf_user_plus_anchors_mmr`. May be empty if no recs.
    """
    steamid64 = str(steamid64).strip()
    if not steamid64:
        raise ValueError("steamid64 must be a non-empty string")

    # --------------------------------------------------
    # 1) Load catalogue + sparse, L2-normalised feature matrix
    df, full_matrix_norm = load_catalogue_and_features()

    # --------------------------------------------------
    # 2) Fetch owned games from Steam
    owned_df = fetch_owned_games(steamid64, api_key)
    if owned_df.empty:
        # No visible games → caller can decide what to do (e.g. CF/popularity).
        return pd.DataFrame()

    # --------------------------------------------------
    # 3) Ensure all owned games exist in the catalogue; if not, extend
    #    catalogue and rebuild the feature matrix BEFORE building
    #    the user profile.
    df, full_matrix_norm = ensure_user_games_in_catalogue_and_refresh(
        owned_df=owned_df,
        catalogue_df=df,
    )

    # After this point, df and full_matrix_norm are guaranteed to include
    # all games from owned_df (if metadata could be fetched).

    # --------------------------------------------------
    # 4) Map owned games to indices in the (potentially updated) catalogue
    owned_mapped = map_owned_to_indices(owned_df, df)
    if owned_mapped.empty:
        return pd.DataFrame()

    # --------------------------------------------------
    # 5) Build a single user content profile vector
    user_vec = build_user_content_profile(
        owned_mapped=owned_mapped,
        full_matrix_norm=full_matrix_norm,
        min_playtime=min_playtime,
    )

    if user_vec is None:
        # Degenerate / cold-start case → nothing to return.
        return pd.DataFrame()

    # --------------------------------------------------
    # 6) Generate CBF recommendations using:
    #    - global user vector
    #    - anchor-based soft similarity
    #    - MMR diversity re-ranking
    recs = recommend_cbf_user_plus_anchors_mmr(
        catalogue_df=df,
        full_matrix_norm=full_matrix_norm,
        owned_mapped=owned_mapped,
        user_vec=user_vec,
        top_n=top_n,
        candidate_pool_size=candidate_pool_size,
        min_playtime=min_playtime,
        beta=beta_anchor_blend,
        lambda_mmr=lambda_mmr,
    )

    # Ensure we always return a DataFrame
    if recs is None:
        return pd.DataFrame()
    return recs.reset_index(drop=True)
