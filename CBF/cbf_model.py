import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import scipy.sparse as sp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize, MultiLabelBinarizer

from scipy.sparse import save_npz, load_npz

# ======================================================
# CONFIG

DATA_DIR = Path("data")
INPUT_CSV = DATA_DIR / "raw" / "game_details.csv"

OUTPUT_DIR = DATA_DIR / "processed"
FEATURE_MATRIX_NPZ = OUTPUT_DIR / "recommender_matrix.npz"

# Max features for TF-IDF blocks
TAG_TFIDF_MAX_FEATURES = 15_000
DESC_TFIDF_MAX_FEATURES = 7_500

# Feature block scaling
TAG_SCALE = 0.9     # genres + tags (strongest signal)
TITLE_SCALE = 0.25     # title TF-IDF
DESC_SCALE = 0.2   # description
DEV_SCALE = 0.2     # developers
PUB_SCALE = 0.1       # publishers



# ======================================================
# TEXT HELPERS

def split_and_normalize_tags(text: str):
    """
    Split semi-structured genres/tags field into clean tokens.

    NOTE:
    - Explicitly blacklists the 'Free to Play' tag (and variants):
      'Free to Play', 'free-to-play', 'FREE TO PLAY', 'f2p', etc.
    - This prevents 'Free to Play' from dominating the TF–IDF tag space,
      and removes all 1/2/3-grams that would originate from that tag.
    """
    if not isinstance(text, str):
        return []

    parts = [p.strip() for p in text.split(";") if p.strip()]
    tokens = []

    for raw in parts:
        raw_lower = raw.lower().strip()

        # --- BLACKLIST: "Free to Play" & variants ---
        # This catches:
        # - "free to play"
        # - "free-to-play"
        # - "free  to   play" (extra spaces)
        # - "f2p"
        if (
            re.fullmatch(r"free\s*to\s*play", raw_lower)
            or raw_lower == "free-to-play"
            or raw_lower == "f2p"
        ):
            # Skip this tag/genre entirely
            continue

        # Normal processing for all other tags/genres
        p = raw_lower
        p = re.sub(r"-+", " ", p)  # hyphens → spaces

        # sometimes tags contain commas, split them too
        comma_split = [x.strip() for x in p.split(",") if x.strip()]

        for chunk in comma_split:
            chunk = re.sub(r"[^a-z0-9 ]+", " ", chunk)
            chunk = re.sub(r"\s+", " ", chunk).strip()
            if chunk:
                tokens.extend(chunk.split(" "))

    return tokens


def split_multi_value_company(text: str):
    """Normalize developer/publisher names into tokens."""
    if not isinstance(text, str):
        return []

    text = text.strip().strip('"').strip("'")
    if not text:
        return []

    if ";" in text:
        parts = [p.strip() for p in text.split(";") if p.strip()]
    elif " / " in text:
        parts = [p.strip() for p in text.split("/") if p.strip()]
    else:
        parts = [text]

    cleaned = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip().lower()
        p = re.sub(r"[^a-z0-9]+", "", p)
        if p:
            cleaned.append(p)

    return cleaned


def desc_tokenizer(text: str):
    """Tokenizer for free-form descriptions."""
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"-+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split(" ") if text else []


# ======================================================
# FEATURE BUILDING

def build_feature_matrix(df: pd.DataFrame) -> sp.csr_matrix:
    """
    Builds a single concatenated TF–IDF + OHE feature matrix for all games.

    RETURNS:
        full_matrix_norm: CSR sparse matrix, shape (n_games, d),
                          with **each row L2-normalised** → final game embedding f_i.
    """
    required = ["title", "genres", "tags", "description", "developers", "publishers"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column in CSV: {col}")

    df = df.copy()

    # ---------- TITLE TF-IDF ----------
    def title_tokenizer(x: str):
        if not isinstance(x, str):
            return []
        x = x.lower()
        x = re.sub(r"-+", " ", x)
        x = re.sub(r"[^a-z0-9 ]+", " ", x)
        x = re.sub(r"\s+", " ", x).strip()
        return x.split(" ") if x else []

    title_vec = TfidfVectorizer(
        tokenizer=title_tokenizer,
        preprocessor=None,
        stop_words="english",
        token_pattern=None,
        ngram_range=(1, 2),
        max_features=7_500,
        sublinear_tf=True,
    )

    tfidf_title = title_vec.fit_transform(df["title"].fillna(""))
    tfidf_title *= TITLE_SCALE
    print("TF–IDF title:", tfidf_title.shape)

    # ---------- GENRES + TAGS TF-IDF ----------
    df["genres_tokens"] = df["genres"].fillna("").apply(split_and_normalize_tags)
    df["tags_tokens"] = df["tags"].fillna("").apply(split_and_normalize_tags)
    df["tag_text"] = (df["genres_tokens"] + df["tags_tokens"]).apply(lambda x: " ".join(x))

    tag_vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=TAG_TFIDF_MAX_FEATURES,
    )

    tfidf_tags = tag_vec.fit_transform(df["tag_text"])
    tfidf_tags *= TAG_SCALE
    print("TF–IDF tags:", tfidf_tags.shape)

    # ---------- DESCRIPTION TF-IDF ----------
    desc_vec = TfidfVectorizer(
        tokenizer=desc_tokenizer,
        preprocessor=None,
        stop_words="english",
        token_pattern=None,
        ngram_range=(1, 2),
        max_features=DESC_TFIDF_MAX_FEATURES,
        min_df=5,
        max_df=0.2,
        sublinear_tf=True,
    )

    tfidf_desc = desc_vec.fit_transform(df["description"].fillna(""))
    tfidf_desc *= DESC_SCALE
    print("TF–IDF description:", tfidf_desc.shape)

    # ---------- DEVELOPER + PUBLISHER OHE ----------
    dev_lists = df["developers"].fillna("").apply(split_multi_value_company)
    pub_lists = df["publishers"].fillna("").apply(split_multi_value_company)

    dev_ohe = MultiLabelBinarizer(sparse_output=True).fit_transform(dev_lists)
    pub_ohe = MultiLabelBinarizer(sparse_output=True).fit_transform(pub_lists)

    dev_ohe = dev_ohe * DEV_SCALE
    pub_ohe = pub_ohe * PUB_SCALE

    print("OHE developer:", dev_ohe.shape)
    print("OHE publisher:", pub_ohe.shape)

    # ---------- COMBINE ALL ----------
    full_matrix = sp.hstack(
        [tfidf_title, tfidf_tags, tfidf_desc, dev_ohe, pub_ohe],
        format="csr",
    )

    print("Full feature matrix:", full_matrix.shape)

    # ---------- L2-NORMALISE EACH ROW ----------
    full_matrix_norm = normalize(full_matrix)
    print("Normalised matrix:", full_matrix_norm.shape)

    # Important: each row is a **single game embedding f_i** with ||f_i||=1.
    return full_matrix_norm


# ======================================================
# LOAD + CACHE

def load_catalogue_and_features(
    csv_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, sp.csr_matrix]:
    """
    Loads the game catalogue and the L2-normalised feature matrix.

    If cached NPZ exists → load directly.
    Else → build and cache to disk.

    RETURNS:
        df: Catalogue DataFrame
        full_matrix_norm: CSR sparse matrix (n_games, d)
    """
    if csv_path is None:
        csv_path = INPUT_CSV

    df = pd.read_csv(csv_path)
    print("Loaded catalogue:", df.shape)

    if "appid" not in df.columns:
        raise ValueError("CSV must contain 'appid' column.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if FEATURE_MATRIX_NPZ.exists():
        print(f"Loading cached feature matrix: {FEATURE_MATRIX_NPZ}")
        full_matrix_norm = load_npz(FEATURE_MATRIX_NPZ)
        print("Loaded:", full_matrix_norm.shape)

    else:
        print("No cached features found → building from scratch…")
        full_matrix_norm = build_feature_matrix(df)
        save_npz(FEATURE_MATRIX_NPZ, full_matrix_norm)
        print("Saved matrix to:", FEATURE_MATRIX_NPZ)

    return df, full_matrix_norm
