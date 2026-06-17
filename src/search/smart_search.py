"""
SmartSearchEngine
=================
Two search modes for the Spotify AI Recommender:

  1. TEXT SEARCH  — TF-IDF cosine similarity on (name + artists) corpus.
                    Falls back to rapidfuzz token-sort ratio when TF-IDF
                    scores are all too low (handles typos / partial names).

  2. VIBE SEARCH  — Natural-language mood queries like "chill night drive"
                    or "gym workout" are mapped to target audio-feature
                    vectors, which are then queried against the FAISS index.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process as rfprocess, fuzz

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vibe → Audio Feature Targets
# ---------------------------------------------------------------------------
# Each key is a tuple of trigger keywords.
# Values are (mean, std) dicts for audio features so we can sample a target
# vector and score tracks near it via FAISS.
# Features: danceability, energy, valence, tempo, speechiness,
#            acousticness, instrumentalness, liveness, popularity
# ---------------------------------------------------------------------------

# All values are in Z-SCORE units (0=average, +1=one std dev above average, -1=below)
# Features: danceability, energy, valence, tempo, speechiness,
#            acousticness, instrumentalness, liveness, popularity
VIBE_PROFILES: dict[tuple, dict] = {
    # ── Gym / workout  (very high energy, fast, electric) ─────────────────
    ("gym", "workout", "running", "hype", "pump", "beast", "intense",
     "aggressive", "power", "training", "cardio", "crossfit"): {
        "danceability":     1.0,   # above-avg danceable
        "energy":           2.0,   # very high energy
        "valence":          0.5,   # positive
        "tempo":            1.5,   # very fast
        "speechiness":      0.5,   # some rap/lyrics
        "acousticness":    -1.5,   # electric
        "instrumentalness": -1.0,  # vocals present
        "liveness":         0.3,
        "popularity":       0.5,
    },

    # ── Party / club  (max danceability, high energy, euphoric) ───────────
    ("party", "club", "dance", "turn up", "disco", "rave", "edm",
     "nightclub", "festival", "banger"): {
        "danceability":     2.0,   # extremely danceable
        "energy":           1.5,
        "valence":          1.5,   # euphoric
        "tempo":            1.0,
        "speechiness":      0.3,
        "acousticness":    -1.5,
        "instrumentalness": -1.0,
        "liveness":         0.5,
        "popularity":       0.8,
    },

    # ── Chill / night drive  (low energy, acoustic, relaxed) ──────────────
    ("chill", "chillout", "relax", "night drive", "lofi", "lo-fi",
     "mellow", "laid back", "evening", "calm", "smooth", "vibes",
     "late night", "drive", "cruise", "sunday"): {
        "danceability":     0.0,
        "energy":          -1.2,   # low energy
        "valence":          0.3,
        "tempo":           -1.0,   # slow
        "speechiness":     -0.5,
        "acousticness":     1.0,   # acoustic
        "instrumentalness": 0.5,
        "liveness":        -0.5,
        "popularity":       0.3,
    },

    # ── Focus / study  (instrumental, very low energy) ────────────────────
    ("focus", "study", "coding", "concentrate", "work", "productivity",
     "reading", "deep work", "ambient", "background"): {
        "danceability":    -0.8,
        "energy":          -1.5,
        "valence":          0.0,
        "tempo":           -1.2,
        "speechiness":     -1.0,   # no lyrics
        "acousticness":     1.2,
        "instrumentalness": 2.0,   # very instrumental
        "liveness":        -0.8,
        "popularity":       0.0,
    },

    # ── Sad / heartbreak  (very low valence, slow, acoustic) ──────────────
    ("sad", "sadness", "heartbreak", "crying", "depressed", "breakup",
     "lonely", "melancholy", "emotional", "hurt", "miss", "missing",
     "tearful", "grief"): {
        "danceability":    -0.8,
        "energy":          -1.5,
        "valence":         -2.0,   # very sad
        "tempo":           -1.2,
        "speechiness":     -0.3,
        "acousticness":     1.2,
        "instrumentalness": -0.3,
        "liveness":        -0.5,
        "popularity":       0.3,
    },

    # ── Romantic / date night  (medium energy, acoustic, warm) ────────────
    ("romantic", "romance", "love", "date", "date night", "dinner",
     "anniversary", "slow dance", "intimate", "couples"): {
        "danceability":     0.0,
        "energy":          -0.8,
        "valence":          0.5,
        "tempo":           -1.0,
        "speechiness":     -0.5,
        "acousticness":     0.8,
        "instrumentalness": -0.3,
        "liveness":        -0.5,
        "popularity":       0.5,
    },

    # ── Happy / feel-good  (high valence, upbeat) ─────────────────────────
    ("happy", "feel good", "upbeat", "joy", "joyful", "good vibes",
     "positive", "sunshine", "bright", "cheerful", "fun"): {
        "danceability":     1.0,
        "energy":           0.8,
        "valence":          2.0,   # very positive / happy
        "tempo":            0.5,
        "speechiness":      0.0,
        "acousticness":    -0.5,
        "instrumentalness": -0.5,
        "liveness":         0.3,
        "popularity":       0.8,
    },

    # ── Acoustic / unplugged  (very acoustic, raw, folk-like) ─────────────
    ("acoustic", "unplugged", "guitar", "folk", "singer songwriter",
     "indie folk", "campfire", "raw", "stripped"): {
        "danceability":    -0.3,
        "energy":          -1.2,
        "valence":          0.3,
        "tempo":           -0.8,
        "speechiness":     -0.5,
        "acousticness":     2.0,   # extremely acoustic
        "instrumentalness": 0.3,
        "liveness":         0.0,
        "popularity":       0.0,
    },

    # ── Motivation / hustle  (high energy, positive, rap/lyrics) ──────────
    ("motivation", "motivational", "inspiring", "inspirational",
     "hustle", "grind", "success", "boss", "confidence"): {
        "danceability":     0.8,
        "energy":           1.5,
        "valence":          0.8,
        "tempo":            1.0,
        "speechiness":      0.8,   # rap / motivational lyrics
        "acousticness":    -1.2,
        "instrumentalness": -0.8,
        "liveness":         0.5,
        "popularity":       0.5,
    },

    # ── Sleep / meditation  (very low energy, very instrumental) ──────────
    ("sleep", "sleeping", "meditation", "meditate", "relax deep",
     "spa", "peace", "peaceful", "tranquil", "zen", "breathe"): {
        "danceability":    -1.5,
        "energy":          -2.0,   # extremely low energy
        "valence":          0.0,
        "tempo":           -1.5,
        "speechiness":     -1.0,
        "acousticness":     1.5,
        "instrumentalness": 2.0,   # fully instrumental
        "liveness":        -1.0,
        "popularity":      -0.5,
    },
}

# Flat keyword → profile target lookup built once at module load
_KEYWORD_TO_PROFILE: dict[str, dict] = {}
for keywords, profile in VIBE_PROFILES.items():
    for kw in keywords:
        _KEYWORD_TO_PROFILE[kw] = profile


def _detect_vibe(query: str) -> Optional[dict]:
    """
    Returns the best-matching audio-feature target dict if the query
    looks like a vibe/mood query, else None.

    Strategy:
      1. Direct keyword match (fastest)
      2. Fuzzy keyword match with rapidfuzz (handles 'chil', 'wurkout', etc.)
    """
    q_lower = query.lower().strip()

    # 1. Direct exact keyword check
    for kw, profile in _KEYWORD_TO_PROFILE.items():
        if kw in q_lower:
            logger.info(f"[VIBE] Exact keyword match: '{kw}'")
            return profile

    # 2. Fuzzy keyword match
    all_keywords = list(_KEYWORD_TO_PROFILE.keys())
    # Use token_sort_ratio so "workout gym" also matches "gym workout"
    best_match = rfprocess.extractOne(
        q_lower, all_keywords,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=80   # only accept ≥80% similarity
    )
    if best_match:
        matched_kw = best_match[0]
        logger.info(f"[VIBE] Fuzzy keyword match: '{matched_kw}' (score={best_match[1]:.1f})")
        return _KEYWORD_TO_PROFILE[matched_kw]

    return None


class SmartSearchEngine:
    """
    Wraps both search modes and exposes a single `search()` method.

    Usage
    -----
    engine = SmartSearchEngine(df, feature_columns, faiss_mean, faiss_std, faiss_index)
    results = engine.search("blinding lights", top_k=8)
    results = engine.search("chill night drive", top_k=8)
    """

    # TF-IDF min score to trust the result directly (no fuzzy fallback needed)
    TFIDF_CONFIDENCE_THRESHOLD = 0.10

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        faiss_mean: np.ndarray,
        faiss_std: np.ndarray,
        faiss_index,          # faiss.Index object
    ):
        self.df = df
        self.feature_columns = feature_columns
        self.faiss_mean = faiss_mean
        self.faiss_std = faiss_std
        self.faiss_index = faiss_index

        # Build TF-IDF index once
        logger.info("[SmartSearch] Building TF-IDF index …")
        self._build_tfidf_index()
        logger.info(f"[SmartSearch] TF-IDF ready. Corpus size: {len(self.df):,}")

    # ------------------------------------------------------------------
    # Internal: TF-IDF
    # ------------------------------------------------------------------

    def _build_tfidf_index(self):
        """Vectorise 'name + artists' for every track."""
        # Safely get artists column (could be 'artists' or 'artist_names')
        artist_col = "artists" if "artists" in self.df.columns else (
            "artist_names" if "artist_names" in self.df.columns else None
        )

        if artist_col:
            corpus = (
                self.df["name"].fillna("").astype(str)
                + " "
                + self.df[artist_col].fillna("").astype(str)
            ).str.lower()
        else:
            corpus = self.df["name"].fillna("").astype(str).str.lower()

        self._corpus = corpus.tolist()

        self._tfidf = TfidfVectorizer(
            analyzer="char_wb",   # character n-grams → handles partial words & typos
            ngram_range=(2, 4),
            min_df=1,
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._tfidf.fit_transform(corpus)

    def _tfidf_search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """Return top_k rows from df ranked by TF-IDF cosine similarity."""
        q_vec = self._tfidf.transform([query.lower()])
        sims = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]
        results = self.df.iloc[top_idx].copy()
        results["_search_score"] = sims[top_idx]
        return results

    def _fuzzy_search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """Rapidfuzz token_sort_ratio on track name corpus (typo-tolerant)."""
        matches = rfprocess.extract(
            query.lower(),
            self._corpus,
            scorer=fuzz.token_sort_ratio,
            limit=top_k,
        )
        # matches = [(string, score, index), ...]
        indices = [m[2] for m in matches]
        scores  = [m[1] / 100.0 for m in matches]
        results = self.df.iloc[indices].copy()
        results["_search_score"] = scores
        return results

    # ------------------------------------------------------------------
    # Internal: Vibe / FAISS
    # ------------------------------------------------------------------

    def _vibe_search(self, target_profile: dict, top_k: int) -> pd.DataFrame:
        """
        Build a target audio-feature vector from the mood profile,
        query FAISS, and return top_k tracks.

        NOTE: self.df features are z-score normalized (mean≈0, std≈1).
        We normalize the target profile the same way before querying FAISS.
        Scoring: 80% FAISS similarity + 20% normalized popularity.
        """
        # Build raw target from profile (values in natural units)
        target_raw = np.array(
            [target_profile.get(f, 0.0) for f in self.feature_columns],
            dtype=np.float32,
        ).reshape(1, -1)

        # Normalize exactly as the index was built
        target_norm = (target_raw - self.faiss_mean) / self.faiss_std

        # Large candidate pool so audio similarity, not popularity, drives ranking
        k_candidates = min(top_k * 20, len(self.df))
        distances, indices = self.faiss_index.search(target_norm, k_candidates)

        results = self.df.iloc[indices[0]].copy()
        results["_faiss_score"] = distances[0]

        # Normalize faiss scores to [0, 1] range for stable weighting
        fs = results["_faiss_score"].values
        fs_min, fs_max = fs.min(), fs.max()
        if fs_max > fs_min:
            fs_norm = (fs - fs_min) / (fs_max - fs_min)
        else:
            fs_norm = np.ones(len(fs))

        # popularity column is already z-scored; shift to [0,1] range
        pop_col = results["popularity"].values
        pop_min, pop_max = pop_col.min(), pop_col.max()
        if pop_max > pop_min:
            pop_norm = (pop_col - pop_min) / (pop_max - pop_min)
        else:
            pop_norm = np.ones(len(pop_col))

        results["_search_score"] = fs_norm * 0.80 + pop_norm * 0.20

        return results.sort_values("_search_score", ascending=False).head(top_k)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 8,
    ) -> tuple[pd.DataFrame, str, str]:
        """
        Main entry point.

        Returns
        -------
        (results_df, search_mode, detected_label)

        search_mode  : "vibe" | "tfidf" | "fuzzy"
        detected_label : human-readable description of what was detected
        """
        query = query.strip()
        if not query:
            return pd.DataFrame(), "none", "Empty query"

        # ── 1. Vibe / mood detection ──────────────────────────────────
        vibe_profile = _detect_vibe(query)
        if vibe_profile:
            label = self._vibe_label(query)
            results = self._vibe_search(vibe_profile, top_k=top_k)
            return results, "vibe", label

        # ── 2. TF-IDF text search ─────────────────────────────────────
        tfidf_results = self._tfidf_search(query, top_k=top_k * 2)
        best_score = tfidf_results["_search_score"].max()

        if best_score >= self.TFIDF_CONFIDENCE_THRESHOLD:
            top = tfidf_results.head(top_k)
            seed = top.iloc[0]
            label = f"**{seed['name']}**"
            if "artists" in seed:
                label += f" by {seed['artists']}"
            return top, "tfidf", label

        # ── 3. Fuzzy fallback ─────────────────────────────────────────
        fuzzy_results = self._fuzzy_search(query, top_k=top_k)
        seed = fuzzy_results.iloc[0]
        label = f"**{seed['name']}**"
        if "artists" in seed:
            label += f" by {seed['artists']}"
        return fuzzy_results, "fuzzy", label

    def get_seed_and_recommend(
        self,
        query: str,
        retriever,           # SpotifyFAISSRetriever instance
        top_k: int = 8,
    ) -> tuple[Optional[pd.Series], pd.DataFrame, str, str]:
        """
        Higher-level helper used by the Streamlit UI.

        For vibe queries  → returns (None, vibe_results, "vibe", label)
        For text queries  → returns (seed_row, faiss_recs, mode, label)
        """
        results, mode, label = self.search(query, top_k=top_k)

        if results.empty:
            return None, pd.DataFrame(), mode, label

        if mode == "vibe":
            # Add final_score column so UI rendering works uniformly
            if "final_score" not in results.columns:
                results = results.copy()
                results["final_score"] = results["_search_score"]
            return None, results, mode, label

        # Text mode: use best matching track as seed for FAISS
        seed_track = results.iloc[0]
        track_id = seed_track.get("id")
        if track_id is None or track_id not in retriever.df["id"].values:
            # fallback: return TF-IDF/fuzzy results directly
            results = results.copy()
            results["final_score"] = results["_search_score"]
            return seed_track, results, mode, label

        faiss_recs = retriever.recommend(track_id, top_k=top_k)
        return seed_track, faiss_recs, mode, label

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _vibe_label(query: str) -> str:
        """Generate a human-readable label for the detected vibe."""
        q = query.strip().title()
        return f"Vibe: *{q}*"
