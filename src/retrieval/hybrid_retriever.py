"""
Hybrid Retriever
=================
Uses Two-Tower (self-supervised) item embeddings for retrieval,
then applies hybrid reranking:

    final_score = α · cosine_sim
                + β · popularity_score
                + γ · diversity_bonus

Diversity bonus: penalizes artists that already appear in the
top-K result set, so each result page has a variety of artists.

Falls back transparently to the original audio-feature FAISS index
if the Two-Tower index has not been built yet.
"""

import pickle
import logging
from typing import Optional, Dict

import numpy as np
import faiss
import pandas as pd
from pathlib import Path

log = logging.getLogger("HybridRetriever")

# ── Paths ─────────────────────────────────────────────────────────────────────
_ORIG_META  = Path("src/retrieval/metadata.pkl")        # raw-feature FAISS
_TT_INDEX   = Path("src/retrieval/tt_faiss.index")      # Two-Tower FAISS
_TT_META    = Path("src/retrieval/tt_metadata.pkl")     # Two-Tower metadata
_ENCODER    = Path("models/item_tower")                 # SavedModel


class HybridRetriever:
    """
    Retriever that combines Two-Tower embeddings with popularity
    and artist-diversity signals.

    Parameters
    ----------
    alpha : float
        Weight for Two-Tower cosine similarity  (default 0.70)
    beta  : float
        Weight for normalised popularity score  (default 0.20)
    gamma : float
        Weight for artist diversity bonus       (default 0.10)
    candidate_multiplier : int
        How many extra candidates to fetch before reranking (default 10×)
    """

    def __init__(
        self,
        alpha: float = 0.70,
        beta:  float = 0.20,
        gamma: float = 0.10,
        candidate_multiplier: int = 10,
    ):
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.k_mul = candidate_multiplier
        self._ready = False

    # ── Load ──────────────────────────────────────────────────────────────────
    def load(self) -> bool:
        """Try to load the Two-Tower index. Returns True on success.
        
        Falls back gracefully if the index has not been built.
        
        Returns
        -------
        bool
            True if successfully loaded, False if index not found or corrupted.
        """
        if not _TT_INDEX.exists() or not _TT_META.exists():
            log.warning(
                "Two-Tower index not found. "
                "Run `poetry run python src/models/train_item_tower.py` first."
            )
            return False

        log.info("Loading Two-Tower FAISS index…")
        try:
            self.index = faiss.read_index(str(_TT_INDEX))
        except Exception as e:
            log.error(f"Failed to load FAISS index from {_TT_INDEX}: {e}")
            return False

        try:
            with open(_TT_META, "rb") as f:
                meta = pickle.load(f)
        except Exception as e:
            log.error(f"Failed to load metadata from {_TT_META}: {e}")
            return False

        self.df          = meta["df"].reset_index(drop=True)
        self.embeddings  = meta["embeddings"].astype(np.float32)   # [N, D]
        self.feature_cols = meta["feature_cols"]

        # Build track-id → row-index lookup
        if "id" in self.df.columns:
            self._id2idx = {tid: i for i, tid in enumerate(self.df["id"])}
        else:
            self._id2idx = {}

        # Pre-compute normalised popularity for fast scoring
        if "popularity" in self.df.columns:
            pop = self.df["popularity"].values.astype(np.float32)
            # popularity column is z-scored; shift to [0,1]
            p_min, p_max = pop.min(), pop.max()
            self._pop_norm = (pop - p_min) / (p_max - p_min + 1e-8)
        else:
            self._pop_norm = np.zeros(len(self.df), dtype=np.float32)

        self._ready = True
        log.info(f"Hybrid retriever ready: {self.index.ntotal:,} Two-Tower vectors")
        return True

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Recommend by Track ID ─────────────────────────────────────────────────
    def recommend_by_id(
        self,
        track_id: str,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Return top_k recommendations for a given track id using hybrid scoring.
        
        Parameters
        ----------
        track_id : str
            Spotify track ID to find similar tracks for.
        top_k : int, optional
            Number of recommendations to return (default 10).
        
        Returns
        -------
        pd.DataFrame
            Top-k recommended tracks with scores and explanations.
        
        Raises
        ------
        KeyError
            If track_id not in the index.
        
        Examples
        --------
        >>> retriever = HybridRetriever()
        >>> retriever.load()
        >>> results = retriever.recommend_by_id("4cOdkLwLK6i33zt0B3GKDA", top_k=5)
        >>> print(results[["name", "artists", "final_score"]])
        """
        if track_id not in self._id2idx:
            raise KeyError(f"Track ID '{track_id}' not found in Two-Tower index.")

        idx  = self._id2idx[track_id]
        qvec = self.embeddings[idx : idx + 1]          # [1, D]
        return self._hybrid_recommend(qvec, exclude_idx=idx, top_k=top_k)

    # ── Recommend by Embedding ────────────────────────────────────────────────
    def recommend_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        exclude_idx: int = -1,
    ) -> pd.DataFrame:
        """Return top_k recommendations for an arbitrary embedding vector.
        
        Parameters
        ----------
        query_embedding : np.ndarray
            Query embedding vector [D] to find similar items for.
        top_k : int, optional
            Number of recommendations (default 10).
        exclude_idx : int, optional
            Row index to exclude from results (default -1, none).
        
        Returns
        -------
        pd.DataFrame
            Top-k tracks with hybrid scores.
        """
        qvec = query_embedding.reshape(1, -1).astype(np.float32)
        return self._hybrid_recommend(qvec, exclude_idx=exclude_idx, top_k=top_k)

    # ── Core Hybrid Reranker ──────────────────────────────────────────────────
    def _hybrid_recommend(
        self,
        query_vec: np.ndarray,
        exclude_idx: int,
        top_k: int,
    ) -> pd.DataFrame:
        """
        1. FAISS search for k_mul × top_k candidates
        2. Compute hybrid score per candidate
        3. Apply diversity re-ranking
        4. Return top_k results
        """
        k_cand = min(top_k * self.k_mul + 1, len(self.df))

        # --- Step 1: FAISS retrieval ---
        sims, idxs = self.index.search(query_vec, k_cand)
        sims = sims[0]    # [k_cand]
        idxs = idxs[0]    # [k_cand]

        # Exclude query track itself
        mask   = idxs != exclude_idx
        idxs   = idxs[mask]
        sims   = sims[mask]

        # Normalise cosine scores to [0, 1]
        s_min, s_max = sims.min(), sims.max()
        if s_max > s_min:
            sims_norm = (sims - s_min) / (s_max - s_min)
        else:
            sims_norm = np.ones_like(sims)

        # --- Step 2: Popularity score ---
        pop_scores = self._pop_norm[idxs]

        # --- Step 3: Base hybrid score (before diversity) ---
        base_score = self.alpha * sims_norm + self.beta * pop_scores

        # --- Step 4: Greedy diversity re-ranking ---
        rows = self.df.iloc[idxs].copy()
        rows["_sim_norm"]   = sims_norm
        rows["_pop_norm"]   = pop_scores
        rows["_base_score"] = base_score
        rows["_orig_idx"]   = idxs
        rows = rows.sort_values("_base_score", ascending=False)

        selected    = []
        artist_seen: dict[str, int] = {}   # artist → count in selected

        for _, row in rows.iterrows():
            artist = str(row.get("artists", row.get("artist_names", "unknown")))

            # Diversity bonus: higher for under-represented artists
            artist_count    = artist_seen.get(artist, 0)
            diversity_bonus = 1.0 / (1.0 + artist_count)   # 1, 0.5, 0.33, …

            final = (
                row["_base_score"]
                + self.gamma * diversity_bonus
            )
            selected.append(
                {
                    **row.to_dict(),
                    "final_score":     round(float(final), 4),
                    "sim_score":       round(float(row["_sim_norm"]), 4),
                    "pop_score":       round(float(row["_pop_norm"]), 4),
                    "diversity_bonus": round(float(diversity_bonus), 4),
                }
            )
            artist_seen[artist] = artist_count + 1

            if len(selected) >= top_k:
                break

        result = pd.DataFrame(selected).drop(
            columns=["_sim_norm", "_pop_norm", "_base_score", "_orig_idx"],
            errors="ignore",
        )
        return result.reset_index(drop=True)

    # ── Fallback: load original FAISS (no Two-Tower) ──────────────────────────
    @staticmethod
    def load_original_retriever():
        """Load the original audio-feature FAISS retriever as fallback."""
        from src.retrieval.faiss_retriever import SpotifyFAISSRetriever
        r = SpotifyFAISSRetriever()
        r.load_index()
        return r


# ── Convenience factory ───────────────────────────────────────────────────────
def get_retriever(alpha: float = 0.70, beta: float = 0.20, gamma: float = 0.10):
    """
    Return a HybridRetriever if the Two-Tower index exists,
    otherwise return the original SpotifyFAISSRetriever.
    """
    hr = HybridRetriever(alpha=alpha, beta=beta, gamma=gamma)
    if hr.load():
        log.info("Using Two-Tower Hybrid Retriever")
        return hr
    log.info("Falling back to original FAISS retriever")
    return HybridRetriever.load_original_retriever()
