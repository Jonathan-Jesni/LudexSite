from pathlib import Path
from typing import Tuple, List, Dict
import time
import random
import json
import re

import requests
import pandas as pd
import scipy.sparse as sp
from bs4 import BeautifulSoup
from requests.exceptions import RequestException

from .cbf_model import (
    INPUT_CSV,            # data/raw/game_details.csv (Path)
    FEATURE_MATRIX_NPZ,   # data/processed/recommender_matrix.npz (Path)
    build_feature_matrix,
)

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
DATA_DIR = INPUT_CSV.parent  # typically data/raw
FEATURE_CACHE = FEATURE_MATRIX_NPZ

REQUEST_DELAY = 0.6  # polite delay between requests
MAX_RETRIES = 4
STORE_API = "https://store.steampowered.com/api/appdetails"
STORE_PAGE = "https://store.steampowered.com/app/{appid}"
USER_AGENTS = [
    # A short rotating list — expand if you like
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/16.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/116.0.0.0 Safari/537.36",
]

COMMON_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://store.steampowered.com/",
}

# cookies to try to bypass simple age-checks (no browser required)
AGE_COOKIES = {
    "birthtime": "189345601",  # sufficiently old
    "mature_content": "1",
    "wants_mature_content": "1",
    "lastagecheckage": "1-January-1980",
}

# -------------------------------------------------------------------
# Helpers: HTTP + retries
# -------------------------------------------------------------------
def ensure_directories():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[catalogue_update] ensured folder: {DATA_DIR}")


def random_headers() -> Dict[str, str]:
    ua = random.choice(USER_AGENTS)
    h = COMMON_HEADERS.copy()
    h["User-Agent"] = ua
    return h


def fetch_json_appdetails(appid: int, timeout: int = 15) -> dict:
    """
    Use the Steam Store `appdetails` API which returns JSON for many apps.
    This is the preferred, fast, and stable source.
    """
    params = {"appids": appid, "cc": "us", "l": "english"}
    headers = random_headers()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(STORE_API, params=params, headers=headers, timeout=timeout)
            if r.status_code != 200:
                # transient error, backoff and retry
                time.sleep(REQUEST_DELAY * attempt)
                continue
            js = r.json()
            # return the entry for the appid (may have 'success': False)
            return js.get(str(appid), {})
        except (RequestException, ValueError):
            time.sleep(REQUEST_DELAY * attempt)
            continue
    return {}


def fetch_html_page(appid: int, timeout: int = 20) -> str:
    """
    Fetch the Steam Store page HTML. Attempts to bypass age-checks
    by sending standard cookies (no browser).
    """
    url = STORE_PAGE.format(appid=appid)
    headers = random_headers()
    session = requests.Session()
    session.headers.update(headers)
    session.cookies.update(AGE_COOKIES)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url + "?l=english&cc=us", timeout=timeout, allow_redirects=True)
            text = r.text or ""
            # if Cloudflare or other blocks are detected, try small backoff and retry
            if r.status_code in (429, 503) or "problem fulfilling your request" in text.lower():
                time.sleep(REQUEST_DELAY * attempt + random.random())
                continue
            return text
        except RequestException as e:
            time.sleep(REQUEST_DELAY * attempt)
            continue
    raise RuntimeError(f"Failed to fetch page for appid={appid}")


# -------------------------------------------------------------------
# Parsing helpers (BeautifulSoup)
# -------------------------------------------------------------------
def _clean_text(node):
    if not node:
        return ""
    return node.get_text(separator=" ", strip=True)


def parse_html_to_metadata(appid: int, title_hint: str, html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")

    # Developers
    developers = sorted(set(
        a.get_text(strip=True) for a in soup.select("div.dev_row a") if a.get_text(strip=True)
    ))

    # Publishers & Genres
    publishers = []
    genres = []
    for block in soup.select("div.details_block"):
        for b in block.find_all("b"):
            label = b.get_text(strip=True)
            if "Publisher" in label:
                pubs = [a.get_text(strip=True) for a in b.parent.find_all("a")]
                publishers.extend(pubs)
            if "Genre" in label:
                gens = [a.get_text(strip=True) for a in b.parent.find_all("a")]
                genres.extend(gens)

    publishers = sorted(set(publishers))
    genres = sorted(set(genres))

    # Tags
    tags = sorted(set(
        a.get_text(strip=True).rstrip(",")
        for a in soup.select("a.app_tag, a.app_tag_trends")
    ))

    # Description
    desc_el = soup.select_one("#game_area_description")
    if desc_el:
        description = desc_el.get_text(separator=" ", strip=True)
    else:
        snip = soup.select_one("div.game_description_snippet")
        description = snip.get_text(strip=True) if snip else ""

    title = title_hint or ""
    # attempt to extract a better title from the page
    title_el = soup.select_one("div.apphub_AppName")
    if title_el:
        t = title_el.get_text(strip=True)
        if t:
            title = t

    return {
        "appid": int(appid),
        "title": title,
        "developers": "; ".join(developers),
        "publishers": "; ".join(publishers),
        "genres": "; ".join(genres),
        "tags": "; ".join(tags),
        "description": description,
    }


# -------------------------------------------------------------------
# High-level: fetch metadata (API-first, HTML fallback)
# -------------------------------------------------------------------
def fetch_game_metadata(appid: int, title_hint: str = "") -> dict:
    """
    Attempt to fetch rich metadata for an appid:
      1) Try Steam Store API `appdetails`
      2) If API missing required fields, fallback to HTML fetch + parse
    Returns a dict matching the CSV schema used by your CBF pipeline.
    """
    # 1) Try API
    api_entry = fetch_json_appdetails(appid)
    if api_entry and api_entry.get("success"):
        data = api_entry.get("data", {})
        # extract fields if present
        title = data.get("name", title_hint or "")
        developers = data.get("developers", []) or []
        publishers = data.get("publishers", []) or []
        genres_list = []
        tags_list = []

        # some API responses include categories/genres
        if "genres" in data and isinstance(data["genres"], list):
            genres_list = [g.get("description", "") for g in data["genres"] if g.get("description")]
        # tags are often missing in API; we will fallback if tags are empty

        description = ""
        if isinstance(data.get("short_description", ""), str):
            description = data.get("short_description", "")

        # if we have enough metadata (tags or description or genres), accept it
        if (description or genres_list or developers) and len(tags_list) >= 1:
            return {
                "appid": int(appid),
                "title": title,
                "developers": "; ".join(developers),
                "publishers": "; ".join(publishers),
                "genres": "; ".join(genres_list),
                "tags": "; ".join(tags_list),
                "description": description,
            }
        # else, fall back to HTML parsing for richer data

    # 2) HTML fallback
    try:
        html = fetch_html_page(appid)
        parsed = parse_html_to_metadata(appid, title_hint, html)
        return parsed
    except Exception as e:
        # Last resort: return a minimal record so pipeline can continue
        print(f"[catalogue_update] fallback failed for {appid}: {e}")
        return {
            "appid": int(appid),
            "title": title_hint or f"appid_{appid}",
            "developers": "",
            "publishers": "",
            "genres": "",
            "tags": "",
            "description": "",
        }


# -------------------------------------------------------------------
# Catalogue extension / rebuild pipeline
# -------------------------------------------------------------------
def extend_catalogue_with_missing_games(
    owned_df: pd.DataFrame,
    catalogue_df: pd.DataFrame,
    max_new: int = 50,
) -> pd.DataFrame:
    """
    Extend catalogue_df with missing appids from owned_df, crawling up to max_new.
    """
    ensure_directories()

    catalog_appids = set(int(a) for a in catalogue_df["appid"].astype(int))
    missing_owned = owned_df[~owned_df["appid"].isin(catalog_appids)].copy()

    if missing_owned.empty:
        print("[catalogue_update] no missing games found.")
        return catalogue_df

    if "playtime_min" in missing_owned.columns:
        missing_owned = missing_owned.sort_values("playtime_min", ascending=False)

    if len(missing_owned) > max_new:
        print(f"[catalogue_update] capping crawls to top {max_new} missing games.")
        missing_owned = missing_owned.head(max_new)

    missing_appids = [int(a) for a in missing_owned["appid"].tolist()]
    title_map = {int(row["appid"]): str(row.get("title", "")).strip()
                 for _, row in owned_df.iterrows()}

    new_rows = []
    for i, appid in enumerate(missing_appids, start=1):
        hint = title_map.get(appid, "")
        print(f"[catalogue_update] ({i}/{len(missing_appids)}) fetching {appid} …")
        try:
            time.sleep(REQUEST_DELAY + random.random() * 0.4)
            row = fetch_game_metadata(appid, title_hint=hint)
            new_rows.append(row)
        except Exception as e:
            print(f"[catalogue_update] error fetching {appid}: {e}")
            continue

    if not new_rows:
        print("[catalogue_update] no new rows fetched.")
        return catalogue_df

    new_df = pd.DataFrame(new_rows)
    merged_df = pd.concat(
        [catalogue_df, new_df.reindex(columns=catalogue_df.columns, fill_value=pd.NA)],
        ignore_index=True,
    )
    print(f"[catalogue_update] extended catalogue by {len(new_df)} rows.")
    return merged_df


def rebuild_feature_matrix_and_cache(updated_df: pd.DataFrame) -> sp.csr_matrix:
    """
    Rebuild the TF–IDF + OHE feature matrix and save to NPZ.
    """
    print("[catalogue_update] rebuilding feature matrix …")
    full_matrix_norm = build_feature_matrix(updated_df)
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    from scipy.sparse import save_npz
    save_npz(FEATURE_CACHE, full_matrix_norm)
    updated_df.to_csv(INPUT_CSV, index=False)
    print(f"[catalogue_update] saved feature matrix -> {FEATURE_CACHE}")
    print(f"[catalogue_update] saved catalogue CSV -> {INPUT_CSV}")
    return full_matrix_norm


def ensure_user_games_in_catalogue_and_refresh(
    owned_df: pd.DataFrame,
    catalogue_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, sp.csr_matrix]:
    """
    Main wrapper: extend catalogue if needed, rebuild matrix when changed.
    """
    original_count = len(catalogue_df)
    updated_df = extend_catalogue_with_missing_games(owned_df, catalogue_df)
    if len(updated_df) == original_count:
        print("[catalogue_update] no change; loading cached feature matrix.")
        from scipy.sparse import load_npz
        full_matrix_norm = load_npz(FEATURE_CACHE)
        return updated_df, full_matrix_norm

    full_matrix_norm = rebuild_feature_matrix_and_cache(updated_df)
    return updated_df, full_matrix_norm
