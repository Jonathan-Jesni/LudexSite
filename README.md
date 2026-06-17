# 🎮 **LudexSite — Hybrid Game Recommendation Web App**

LudexSite is the web application interface for **Ludex**, a fully offline, hybrid **Content-Based + Collaborative Filtering** recommendation engine designed to provide **high-quality, diverse, and personalized Steam game suggestions**. 

Built with Flask, LudexSite allows users to authenticate via Steam OpenID, analyze their gaming history, and receive tailored game recommendations that outperform standard discoverability algorithms by blending metadata, player history, and diversity-aware ranking.

---

## 🧩 **Overview**

The recommendation engine powers the site using three core components:

1. **Content-Based Filtering (CBF)**: Analyzes metadata such as descriptions, tags, genres, and developers using TF-IDF and One-Hot Encoding.
2. **Collaborative Filtering (CF)**: Finds similar users and recommends games based on playtime and ownership behavior patterns.
3. **MMR Re-ranking (Maximal Marginal Relevance)**: Ensures the final recommended list is both highly relevant and diverse.

---

## 🏗️ **Project Structure**

```
LudexSite/
│── CBF/                 # Content-Based Filtering models and logic
│── CF/                  # Collaborative Filtering models and logic
│── data/                # Processed feature matrices and catalogues
│── static/              # CSS, JS, and image assets
│── templates/           # Flask HTML templates
│── app.py               # Main Flask application (Render optimized)
│── app_local.py         # Full-featured Flask application for local testing
│── requirements.txt     # Python dependencies
```

---

## ⚡ **Render Free Tier Optimizations (`app.py`)**

To successfully deploy LudexSite on Render's free tier, which imposes a strict 512MB RAM limit, `app.py` has been heavily optimized compared to its original state. The modifications include:

- **Reduced Candidate Pools:** Decreased `CANDIDATE_POOL_SIZE` from 300 down to 50 to limit the number of items processed in memory simultaneously.
- **Incremental MMR Computation:** Replaced full similarity matrix multiplication (`cand_vecs @ cand_vecs.T`) with incremental dot products in a loop (`row_vec.dot(sel_vecs.T)`) during the MMR re-ranking phase. 
- **Sparse Matrix Handling:** Leveraged `.toarray().ravel()` on sparse matrix slices to resolve `ValueError` exceptions and prevent memory crashes when calculating similarities.
- **Lazy Loading & Concurrency:** Adjusted cache behavior via `get_cbf_data()` and reduced `METADATA_THREADS` from 4 to 2 to minimize parallel memory allocation spikes.

> [!TIP]
> If you are running the project locally and have sufficient RAM, it is recommended to run **`app_local.py`** instead. This version retains the original `CANDIDATE_POOL_SIZE` of 300 and the full matrix similarity operations for more accurate and diverse recommendations.

---

## ▶️ **Running the Project Locally**

### 1. Install Dependencies

Ensure you have Python 3.12+ installed. Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate      # On Linux/Mac
venv\Scripts\activate         # On Windows

pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the `LudexSite` directory and add your Steam API key and secret key:

```env
STEAM_API_KEY=YOUR_STEAM_API_KEY
LUDEX_SECRET=your_secret_key_here
```

### 3. Start the Application

To run the fully-featured local version (recommended):
```bash
python app_local.py
```

To run the Render-optimized version:
```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000`.

---

## 📄 **License**

MIT License  
© 2026 Ludex Project Authors
