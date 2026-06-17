# Moodify Project Report

## 1. Problem Statement

Music apps need recommendations that feel personal, fast, and explainable. A
simple popularity list is easy to build, but it does not adapt to a user's
current mood, favorite artists, or listening context.

Moodify solves this by combining multiple retrieval methods:

- text search for known songs and artists
- fuzzy search for typos and partial queries
- vibe search for natural-language moods
- FAISS similarity search over audio features
- self-supervised item embeddings for richer retrieval
- profile-aware two-tower retrieval using playlist/mood/genre groups
- hybrid reranking with popularity and artist diversity

## 2. Dataset

The project supports two data sources:

- playlist-based data collected through the Spotify API
- a large Spotify-style Kaggle dataset used for scalable retrieval experiments

Important fields include:

- track name and artist
- popularity
- genre or mood label
- audio features such as danceability, energy, valence, tempo, speechiness,
  acousticness, instrumentalness, liveness, and popularity

## 3. Pipeline

```text
Data collection
  -> cleaning and normalization
  -> exploratory data analysis
  -> feature matrix creation
  -> FAISS index creation
  -> optional self-supervised item-tower training
  -> hybrid retrieval and reranking
  -> Streamlit/Flask demo
```

## 4. Recommendation Methods

### Mood Recommendations

For the smaller playlist dataset, songs are grouped by mood labels such as sad,
romantic, party, chill, and old melody. The app can return the most relevant
tracks for each mood.

### Similar Track Recommendations

The system builds a feature vector for each track and retrieves nearby songs
using cosine similarity or FAISS nearest-neighbor search.

### Smart Search

The Streamlit demo uses `SmartSearchEngine` to support:

- TF-IDF character n-gram search for song and artist names
- RapidFuzz fallback for typo-tolerant matching
- natural-language vibe detection for queries like `gym workout` or
  `chill night drive`

### Self-Supervised Item Tower

The item tower creates two noisy augmented views of the same track's audio
features. A contrastive loss trains the model to pull same-track views together
and push different-track views apart. The resulting embeddings are indexed with
FAISS.

### Profile-Aware Two-Tower Model

The profile-aware two-tower script builds a real profile vector from the other tracks in the same
playlist, mood, or genre group, then trains the user tower to retrieve matching
track embeddings from the item tower.

This is still an offline proxy for personalization, but it is defensible because
the training signal comes from real group membership in the dataset.

## 5. Hybrid Reranking

The hybrid retriever combines three signals:

```text
final_score = alpha * similarity + beta * popularity + gamma * diversity
```

This helps avoid a common recommender problem: returning tracks that are all
technically similar but repetitive or dominated by the same artist.

## 6. Evaluation Plan

The included evaluation script reports:

- Precision@K
- Recall@K
- MAP@K
- NDCG@K
- catalog coverage
- artist diversity
- average latency per query

The first version uses mood or genre as proxy relevance labels. A stronger
future version should use real user feedback:

- liked songs
- playlist additions
- skips
- repeat plays
- search-to-play events

Current sample run on `spotify_songs_cleaned.csv`:

| Metric | Value |
| --- | ---: |
| Precision@10 | 0.472 |
| Recall@10 | 0.047 |
| MAP@10 | 0.364 |
| NDCG@10 | 0.491 |
| Catalog coverage | 0.918 |
| Artist diversity | 0.203 |
| Avg latency/query | 2.08 ms |

Interpretation: the recommender retrieves mood-consistent tracks at a useful
rate and covers a large portion of the catalog, but artist diversity is still a
clear improvement area.

## 7. Current Strengths

- Full-stack ML project rather than only a notebook
- Multiple retrieval strategies in one system
- FAISS-based scalable search
- Streamlit UI suitable for demos
- MLflow artifacts and experiment tracking
- Clear path to production evaluation and personalization

## 8. Current Limitations

- Some older scripts still contain prototype logic.
- The profile-aware two-tower model uses playlist, mood, or genre groups
  as proxy users. Real listener-level events would be stronger.
- Offline labels are proxy labels, not real user satisfaction labels.
- The Flask and Streamlit demos should eventually be consolidated into one
  official demo experience.

## 9. Future Work

- Build user profiles from liked playlists and recently played tracks.
- Use real listener-level interactions for two-tower training.
- Deploy the Streamlit demo and add a public demo link.
- Add CI, unit tests, and linting.
- Track model versions and data versions with MLflow/DVC.

## 10. Portfolio Summary

Moodify demonstrates practical recommendation-system engineering: data
preparation, vector retrieval, hybrid ranking, self-supervised embeddings,
experiment tracking, and interactive product presentation.
