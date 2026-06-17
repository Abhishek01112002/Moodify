# Quick Start Guide

This guide gets the project running locally and points you to the most useful
commands for demos, data processing, retrieval, and evaluation.

## 1. Prerequisites

- Python 3.11 is recommended for the full Poetry workflow.
- Python 3.8+ can run the older lightweight scripts.
- Spotify Developer credentials are optional for offline recommendation, but
  required for album art, previews, and Spotify API collection.
- Kaggle credentials are required only if you want to download the large
  Spotify-style dataset.

## 2. Install Dependencies

Recommended:

```bash
poetry install
```

Lightweight legacy pipeline:

```bash
pip install -r requirements.txt
```

## 3. Configure Spotify Credentials

Copy the template:

```bash
copy .env.example .env
```

Fill in:

```text
SPOTIPY_CLIENT_ID=your_spotify_client_id
SPOTIPY_CLIENT_SECRET=your_spotify_client_secret
SPOTIPY_REDIRECT_URI=http://127.0.0.1:8888/callback
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
```

Keep real secrets out of GitHub. Rotate credentials if they were ever committed
or shared.

## 4. Run the Recommended Demo

```bash
poetry run streamlit run app/main.py
```

The Streamlit demo supports:

- song and artist search
- typo-tolerant fuzzy search
- natural-language vibe search
- FAISS recommendations
- optional two-tower hybrid retrieval
- Spotify previews when credentials are available
- recommendation explanations

## 5. Run the Legacy Flask Demo

```bash
python app.py
```

Open:

```text
http://localhost:5000
```

## 6. Playlist-Based Data Pipeline

Use this path for the smaller custom playlist dataset.

```bash
python main.py
python run_phase2_phase3.py
```

Expected outputs:

- `labeled_spotify_songs.csv`
- `spotify_songs_cleaned.csv`
- EDA plots in `eda_output/`

## 7. Large Dataset Pipeline

Use this path for the larger FAISS and item-tower workflow.

```bash
poetry run python src/data_ingestion.py
poetry run python src/preprocessing.py
poetry run python src/retrieval/faiss_retriever.py
```

Expected outputs:

- `data/processed/tracks_large_cleaned.parquet`
- `src/retrieval/faiss_index.index`
- `src/retrieval/metadata.pkl`

## 8. Train the Self-Supervised Item Tower

```bash
poetry run python src/models/train_item_tower.py
```

Expected outputs:

- `models/item_tower/`
- `src/retrieval/tt_faiss.index`
- `src/retrieval/tt_metadata.pkl`

## 9. Train the Profile-Aware Two-Tower Model

```bash
poetry run python src/models/two_tower_model.py --data data/processed/tracks_large_cleaned.parquet --epochs 10
```

This script uses real playlist, mood, or genre groups to create user/profile
vectors from the data.

Expected outputs:

- `models/profile_two_tower/`
- `models/profile_user_tower/`
- `models/profile_item_tower/`
- `src/retrieval/profile_tt_faiss.index`
- `src/retrieval/profile_tt_metadata.pkl`

## 10. Run Evaluation

```bash
python scripts/evaluate_recommender.py --data spotify_songs_cleaned.csv --top-k 10 --sample-size 200
```

To save a JSON report:

```bash
python scripts/evaluate_recommender.py --data spotify_songs_cleaned.csv --top-k 10 --sample-size 200 --output reports/evaluation_sample.json
```

Metrics include:

- Precision@K
- Recall@K
- MAP@K
- NDCG@K
- catalog coverage
- artist diversity
- average latency per query

## 11. Recommended GitHub Publishing Checklist

- Remove or ignore all secret files.
- Add 2-3 screenshots of the Streamlit app.
- Add a 30-60 second demo video.
- Include `reports/evaluation_summary.md` in the repository.
- Mention limitations honestly in `PROJECT_REPORT.md`.
- Deploy the Streamlit app and add the public link to `README.md`.
