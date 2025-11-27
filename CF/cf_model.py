"""
Ludex CF Model Module
---------------------

Analogous to `recommender/model.py` on the CBF side.

Responsibilities:
    - Load & filter the interactions CSV
    - Build the user–item implicit feedback matrix
    - Train ALS
    - Save / load the ALS model and ID index

Public API:
    from cf_model import load_cf_model

    model, user_ids, item_ids = load_cf_model(force_retrain=False)

Behavior:
    - If cf_als_model.pkl and cf_als_index.pkl already exist AND
      force_retrain=False:
        → load them from data/processed and return.
    - If they do NOT exist, or force_retrain=True:
        → train ALS from data/raw/user_game_playtime_top20.csv,
          save model + index into data/processed, and return them.

Author: Ludex Project
"""

from pathlib import Path
import pickle
from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import implicit


# ======================================================
# PATHS

BASE = Path(__file__).resolve().parent.parent

# Models live in data/processed
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

INTERACTIONS_CSV = BASE / "data" / "raw" / "user_game_playtime_top20.csv"

MODEL_PATH = PROC / "cf_als_model.pkl"
INDEX_PATH = PROC / "cf_als_index.pkl"


# ======================================================
# CONFIG

MIN_PLAYTIME = 60          # drop interactions with tiny playtime (minutes)
MIN_USER_SUPPORT = 2       # game must be played by ≥2 users
ALS_FACTORS = 64
ALS_REG = 0.15
ALS_ITERS = 25
RANDOM_STATE = 42


# ======================================================
# LOAD + FILTER

def load_and_filter() -> Tuple[pd.DataFrame, pd.Index, pd.Index]:
    """
    Load the raw interactions CSV and apply all filters.
    Then factorize user and item into contiguous integer indices.

    Returns
    -------
    df : pd.DataFrame
        Interactions with columns [steamid, appid, playtime_forever,
        user_idx, item_idx].
    user_ids : pd.Index
        Mapping user_idx -> SteamID.
    item_ids : pd.Index
        Mapping item_idx -> AppID.
    """
    print(f"\n[Ludex CF] Loading interactions from: {INTERACTIONS_CSV}")

    try:
        df = pd.read_csv(
            INTERACTIONS_CSV,
            usecols=["steamid", "appid", "playtime_forever"],
        )
    except FileNotFoundError:
        raise SystemExit(
            f"Ludex Error: Interactions CSV not found:\n{INTERACTIONS_CSV}"
        )

    if df.empty:
        raise SystemExit("Ludex Error: interactions CSV is empty.")

    # Combine duplicate (user, item)
    df = (
        df.groupby(["steamid", "appid"], as_index=False)["playtime_forever"]
          .sum()
    )

    # Filter: playtime threshold
    df = df[df["playtime_forever"] >= MIN_PLAYTIME]
    print(f"[Ludex CF] After MIN_PLAYTIME={MIN_PLAYTIME}: {len(df)} rows")

    if df.empty:
        raise SystemExit(
            f"Ludex Error: No interactions remain after MIN_PLAYTIME.\n"
            "Check crawler output or reduce filter thresholds."
        )

    # Filter: item must have enough users
    item_counts = df.groupby("appid")["steamid"].nunique()
    valid_items = item_counts[item_counts >= MIN_USER_SUPPORT].index
    df = df[df["appid"].isin(valid_items)]
    print(f"[Ludex CF] After MIN_USER_SUPPORT={MIN_USER_SUPPORT}: {len(df)} rows")

    if df.empty:
        raise SystemExit(
            f"Ludex Error: All items removed after MIN_USER_SUPPORT filter.\n"
            "Reduce MIN_USER_SUPPORT or collect more crawl data."
        )

    df = df.reset_index(drop=True)

    # Factorize IDs → contiguous indices
    df["user_idx"], user_ids = pd.factorize(df["steamid"])
    df["item_idx"], item_ids = pd.factorize(df["appid"])

    n_users = df["user_idx"].nunique()
    n_items = df["item_idx"].nunique()

    print(f"[Ludex CF] Users={n_users}, Items={n_items}, Interactions={len(df)}")

    if n_users < 2 or n_items < 2:
        raise SystemExit(
            f"Ludex Error: Not enough users/items for ALS "
            f"(users={n_users}, items={n_items})."
        )

    return df, user_ids, item_ids


# ======================================================
# MATRIX BUILDING

def build_user_item_matrix(df: pd.DataFrame) -> coo_matrix:
    """
    Construct a user–item implicit matrix using normalized playtime.

    Returns
    -------
    csr_matrix of shape (n_users, n_items)
    """
    # Normalize per-user playtime to [0,1]
    df["norm_playtime"] = df.groupby("user_idx")["playtime_forever"].transform(
        lambda x: x / x.max()
    )

    # Convert to implicit confidence
    confidence = np.log1p(df["norm_playtime"] * 40).astype(np.float32)

    rows = df["user_idx"].astype(np.int32).values
    cols = df["item_idx"].astype(np.int32).values

    n_users = df["user_idx"].nunique()
    n_items = df["item_idx"].nunique()

    matrix = coo_matrix(
        (confidence, (rows, cols)),
        shape=(n_users, n_items),
    ).tocsr()

    return matrix


# ======================================================
# ALS TRAINING

def train_als(user_items: coo_matrix) -> implicit.als.AlternatingLeastSquares:
    """
    Train the implicit ALS model on the given user-items matrix.
    """
    print("\n[Ludex CF] Training implicit ALS model…")

    model = implicit.als.AlternatingLeastSquares(
        factors=ALS_FACTORS,
        regularization=ALS_REG,
        iterations=ALS_ITERS,
        random_state=RANDOM_STATE,
    )

    model.fit(user_items)
    return model


# ======================================================
# PUBLIC API: LOAD / TRAIN + SAVE

def load_cf_model(force_retrain: bool = False):
    """
    High-level helper for CF consumers (e.g., recommend_for_user.py).

    Behavior:
        - If force_retrain is False AND cf_als_model.pkl AND cf_als_index.pkl
          both exist:
              → load and return (model, user_ids, item_ids).
        - Otherwise (no existing model/index OR force_retrain=True):
              → train ALS from CSV, save model+index, then return them.

    Parameters
    ----------
    force_retrain : bool, default False
        If True, ignore any existing saved model/index and retrain from
        the current interactions CSV, overwriting the .pkl files.

    Returns
    -------
    model : implicit.als.AlternatingLeastSquares
    user_ids : list[str]
    item_ids : list[int]
    """
    # Decide whether to load or train
    if (not force_retrain) and MODEL_PATH.exists() and INDEX_PATH.exists():
        print("[Ludex CF] Found existing CF model + index. Loading from disk…")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(INDEX_PATH, "rb") as f:
            idx = pickle.load(f)

        user_ids: List[str] = list(idx["user_ids"])
        item_ids: List[int] = list(idx["item_ids"])
        return model, user_ids, item_ids

    print(
        "[Ludex CF] "
        + ("Force retrain requested." if force_retrain else "No existing CF model found.")
        + " Training from scratch…"
    )

    df, user_ids_idx, item_ids_idx = load_and_filter()
    user_items = build_user_item_matrix(df)
    model = train_als(user_items)

    # Save model + index (overwrite if already present)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump(
            {"user_ids": list(user_ids_idx),
             "item_ids": list(item_ids_idx)},
            f,
        )

    print("[Ludex CF] ✓ Model trained and saved to data/processed.")
    return model, list(user_ids_idx), list(item_ids_idx)
