"""
Ludex CF Interactions Update
----------------------------

Analogous to `recommender/catalogue_update.py` on the CBF side.

Responsibilities:
    - Ensure that a set of SteamIDs exist in the interactions CSV
      (data/raw/user_game_playtime_top20.csv)
    - If new rows are appended, trigger a full CF retrain via cf_model.load_cf_model(force_retrain=True)

Typical usage from recommend_for_user.py:
    from cf_interactions_update import (
        load_interactions,
        ensure_users_in_data_and_retrain,
    )

    df = load_interactions()
    ensure_users_in_data_and_retrain([target_steamid] + friends)

Author: Ludex Project
"""

from pathlib import Path
import os
import json
import time
from typing import Dict, List, Set, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

from CF.cf_model import (
    BASE,
    INTERACTIONS_CSV,
    load_cf_model,     
)


# ======================================================
# PATHS / ENV
# ======================================================

# Load .env from project root (Ludex/.env)
ENV_PATH = BASE / ".env"
load_dotenv(ENV_PATH)

# raw cache (Ludex/data/raw/players)
RAW_PLAYERS = BASE / "data" / "raw" / "players"
RAW_PLAYERS.mkdir(parents=True, exist_ok=True)

# Steam API Key loaded from .env
STEAM_API_KEY = os.getenv("STEAM_API_KEY")

FRIENDS_URL = "https://api.steampowered.com/ISteamUser/GetFriendList/v1/"
OWNED_URL   = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"
REQUEST_DELAY = 0.5  # seconds between live API calls (polite)


# ======================================================
# SMALL UTILITIES (HTTP + OWNED GAMES)

def safe_get(url: str, params: dict, max_retries: int = 3):
    """GET with simple retries + polite backoff."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r
            else:
                print(f"HTTP {r.status_code} for {url} (params={params})")
        except Exception as e:
            print(f"Error on GET {url}: {e}")
        time.sleep(REQUEST_DELAY * (attempt + 1))
    return None


def fetch_owned_games(steamid: str):
    """
    Fetch owned games and ALWAYS use cached JSON if available.
    Only fall back to Steam API if no cache exists AND API key is available.
    This ensures we always know what games the user owns, even without API key.
    """
    # 1) Try quick recommender cache
    out_quick = RAW_PLAYERS / f"{steamid}_owned_quick.json"
    if out_quick.exists():
        try:
            return json.loads(out_quick.read_text())
        except Exception:
            pass

    # 2) Try old crawler cache (<steamid>_owned.json)
    out_old = RAW_PLAYERS / f"{steamid}_owned.json"
    if out_old.exists():
        try:
            return json.loads(out_old.read_text())
        except Exception:
            pass

    # 3) Only if NO cache, attempt Steam API fetch
    if not STEAM_API_KEY:
        print("WARNING: STEAM_API_KEY not set and no cached owned-games JSON.")
        return None

    params = {
        "key": STEAM_API_KEY,
        "steamid": steamid,
        "include_played_free_games": 1,
        "include_appinfo": 0,
    }
    r = safe_get(OWNED_URL, params)
    if not r:
        return None

    js = r.json()
    out_quick.write_text(json.dumps(js, indent=2))
    return js


def has_public_games(owned_json) -> bool:
    if not owned_json:
        return False
    resp = owned_json.get("response", {})
    games = resp.get("games", [])
    return len(games) > 0


def select_top_games_from_json(owned_json, top_n: int = 20):
    """Return list of {'appid', 'playtime_forever'} from owned_json."""
    if not has_public_games(owned_json):
        return []

    resp = owned_json.get("response", {})
    games = resp.get("games", [])
    if not games:
        return []

    games_sorted = sorted(
        games,
        key=lambda g: int(g.get("playtime_forever", 0)),
        reverse=True,
    )
    selected = games_sorted[:top_n]
    rows = []
    for g in selected:
        playtime = int(g.get("playtime_forever", 0))
        if playtime <= 0:
            continue
        rows.append(
            {
                "steamid": str(resp.get("steamid", "")) or str(g.get("steamid", "")),
                "appid": int(g.get("appid")),
                "playtime_forever": playtime,
            }
        )
    return rows


# ======================================================
# INTERACTIONS HELPERS

def load_interactions() -> pd.DataFrame:
    """
    Load the interactions CSV from disk, with basic sanity checks.
    """
    if not INTERACTIONS_CSV.exists():
        raise SystemExit(f"Missing interactions CSV: {INTERACTIONS_CSV}")
    df = pd.read_csv(INTERACTIONS_CSV)
    if df.empty:
        raise SystemExit("ERROR: interactions CSV is empty.")
    df["steamid"] = df["steamid"].astype(str)
    df["appid"] = df["appid"].astype(int)
    return df


def ensure_users_in_data_and_retrain(steamids: List[str]) -> None:
    """
    For each steamid not present in INTERACTIONS_CSV:
      - fetch owned games (top 20, public only)
      - append to CSV

    If any rows were added, retrain ALS once via load_cf_model(force_retrain=True).

    This mirrors the logic of `ensure_user_games_in_catalogue_and_refresh`
    on the CBF side, but for CF interactions instead of game metadata.
    """
    if not STEAM_API_KEY:
        print("STEAM_API_KEY not set; cannot auto-add new users.")
        return

    steamids = [str(s) for s in steamids]
    df = load_interactions()
    existing_users = set(df["steamid"].unique())

    new_rows = []
    for sid in steamids:
        if sid in existing_users:
            continue
        print(f"[CF interactions_update] Adding new user {sid} via Steam API…")
        owned_json = fetch_owned_games(sid)
        rows = select_top_games_from_json(owned_json, top_n=20)
        for r in rows:
            r["steamid"] = sid
        new_rows.extend(rows)

    if not new_rows:
        print("[CF interactions_update] No new playable games; no retrain needed.")
        return

    df_new = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df_new.drop_duplicates(subset=["steamid", "appid"], inplace=True)

    # Overwrite CSV – keeps it as single source of truth
    df_new.to_csv(INTERACTIONS_CSV, index=False)
    print(f"[CF interactions_update] Appended {len(new_rows)} new interaction rows.")

    # Full retrain to incorporate the new users/items
    print("[CF interactions_update] Retraining CF model with updated interactions…")
    load_cf_model(force_retrain=True)
    print("[CF interactions_update] Retrain complete.")
