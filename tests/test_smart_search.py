"""Unit tests for SmartSearchEngine."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.search.smart_search import SmartSearchEngine


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a minimal test dataframe."""
    return pd.DataFrame({
        "name": ["Blinding Lights", "Heat Waves", "Good as Hell"],
        "artists": ["The Weeknd", "Glass Animals", "Lizzo"],
        "danceability": [-0.2, 0.1, 1.2],
        "energy": [0.5, 0.3, 1.0],
        "valence": [0.0, 0.2, 1.5],
        "tempo": [0.8, -0.3, 0.5],
        "speechiness": [-0.5, -0.6, -0.3],
        "acousticness": [-1.2, -0.8, -1.5],
        "instrumentalness": [-1.0, -0.9, -1.2],
        "liveness": [0.2, 0.1, 0.3],
        "popularity": [0.5, 0.3, 0.8],
    })


def test_smart_search_engine_init(sample_df: pd.DataFrame) -> None:
    """Test SmartSearchEngine initialization."""
    feature_cols = ["danceability", "energy", "valence", "tempo", "speechiness",
                    "acousticness", "instrumentalness", "liveness", "popularity"]

    engine = SmartSearchEngine(
        df=sample_df,
        feature_columns=feature_cols,
        faiss_mean=np.zeros(9),
        faiss_std=np.ones(9),
        faiss_index=None,
    )

    assert engine is not None
    assert len(engine.df) == 3
    assert engine.search is not None


def test_text_search(sample_df: pd.DataFrame) -> None:
    """Test text-based search with TF-IDF."""
    feature_cols = ["danceability", "energy", "valence", "tempo", "speechiness",
                    "acousticness", "instrumentalness", "liveness", "popularity"]

    engine = SmartSearchEngine(
        df=sample_df,
        feature_columns=feature_cols,
        faiss_mean=np.zeros(9),
        faiss_std=np.ones(9),
        faiss_index=None,
    )

    results, mode, label = engine.search("Blinding Lights", top_k=1)
    assert len(results) == 1
    assert results.iloc[0]["name"] == "Blinding Lights"
    assert mode in ("tfidf", "fuzzy")


def test_text_search_typo_fallback(sample_df: pd.DataFrame) -> None:
    """Test fuzzy search fallback for typos."""
    feature_cols = ["danceability", "energy", "valence", "tempo", "speechiness",
                    "acousticness", "instrumentalness", "liveness", "popularity"]

    engine = SmartSearchEngine(
        df=sample_df,
        feature_columns=feature_cols,
        faiss_mean=np.zeros(9),
        faiss_std=np.ones(9),
        faiss_index=None,
    )

    results, mode, label = engine.search("blinding", top_k=1)
    assert len(results) >= 1
    assert "Blinding" in results.iloc[0]["name"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
