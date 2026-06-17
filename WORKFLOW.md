# Project Workflow

This document explains how the major parts of the recommendation system connect.

## Overview

```text
Data source
  -> ingestion
  -> cleaning and normalization
  -> EDA
  -> feature matrix
  -> retrieval index
  -> recommendation app
  -> evaluation report
```

The project currently has two workflows:

- playlist workflow: smaller custom dataset collected through Spotify playlists
- large dataset workflow: scalable FAISS and item-tower retrieval experiments
- profile-aware two-tower workflow: trains playlist/mood/genre profile vectors
  from the dataset

## Workflow A: Playlist Dataset

Use this path when collecting your own playlists and mood labels.

### Step 1: Configure credentials

Create `.env` from `.env.example` and add your Spotify credentials.

### Step 2: Collect playlist data

```bash
python main.py
```

This runs the collection pipeline and writes:

```text
labeled_spotify_songs.csv
```

### Step 3: Clean data and run EDA

```bash
python run_phase2_phase3.py
```

This writes:

```text
spotify_songs_cleaned.csv
eda_output/
```

### Step 4: Run the Flask demo

```bash
python app.py
```

Open:

```text
http://localhost:5000
```

## Workflow B: Large Dataset + FAISS

Use this path for the stronger portfolio version of the project.

### Step 1: Download large dataset

```bash
poetry run python src/data_ingestion.py
```

### Step 2: Preprocess

```bash
poetry run python src/preprocessing.py
```

This creates:

```text
data/processed/tracks_large_cleaned.parquet
```

### Step 3: Build FAISS index

```bash
poetry run python src/retrieval/faiss_retriever.py
```

This creates:

```text
src/retrieval/faiss_index.index
src/retrieval/metadata.pkl
```

### Step 4: Train item tower

```bash
poetry run python src/models/train_item_tower.py
```

This creates:

```text
models/item_tower/
src/retrieval/tt_faiss.index
src/retrieval/tt_metadata.pkl
```

### Step 5: Run Streamlit demo

```bash
poetry run streamlit run app/main.py
```

## Workflow C: Profile-Aware Two-Tower

Use this path when you want a stronger personalization story based on playlist,
mood, or genre groups.

```bash
poetry run python src/models/two_tower_model.py --data data/processed/tracks_large_cleaned.parquet --epochs 10
```

The script automatically chooses a profile column in this order:

```text
playlist_id -> playlist_name -> mood_label -> primary_genre -> genres -> track_genre
```

For each track, it builds the user/profile vector from the other tracks in the
same profile group, then trains a two-tower retrieval model.

## Recommendation Flow

```text
User query
  -> SmartSearchEngine
  -> vibe detection or TF-IDF/fuzzy match
  -> seed track or target vibe vector
  -> FAISS retrieval
  -> hybrid reranking
  -> result cards in app
```

## Hybrid Scoring

The two-tower hybrid retriever uses:

```text
final_score = alpha * similarity + beta * popularity + gamma * diversity
```

The Streamlit sidebar allows these weights to be adjusted when the hybrid index
is available.

## Recommendation Explanations

The official Streamlit app now shows "Why this song?" details for each result:

- ranking score
- vector similarity
- popularity signal
- diversity bonus when available
- energy, valence, and danceability cues

## Evaluation Flow

```bash
python scripts/evaluate_recommender.py --data spotify_songs_cleaned.csv --top-k 10 --sample-size 200
```

The current evaluation uses mood or genre as proxy relevance. This is useful for
offline comparison, but a production recommender should use real behavior:

- likes
- playlist saves
- skips
- repeat plays
- search-to-play events

## Best Next Technical Improvements

- Replace proxy profile groups with real listener-level interaction features.
- Add a baseline comparison table to the evaluation report.
- Deploy the Streamlit demo and add the public URL to the README.
