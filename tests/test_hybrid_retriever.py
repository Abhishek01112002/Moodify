"""Unit tests for HybridRetriever scoring logic."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.hybrid_retriever import HybridRetriever


def test_hybrid_retriever_init() -> None:
    """Test HybridRetriever initialization with default parameters."""
    retriever = HybridRetriever()
    
    assert retriever.alpha == 0.70
    assert retriever.beta == 0.20
    assert retriever.gamma == 0.10
    assert retriever.k_mul == 10
    assert not retriever.is_ready


def test_hybrid_retriever_custom_weights() -> None:
    """Test HybridRetriever with custom scoring weights."""
    retriever = HybridRetriever(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        candidate_multiplier=5,
    )
    
    assert retriever.alpha == 0.5
    assert retriever.beta == 0.3
    assert retriever.gamma == 0.2
    assert retriever.k_mul == 5


def test_hybrid_retriever_weights_sum() -> None:
    """Verify hybrid scoring weights are reasonable."""
    retriever = HybridRetriever()
    total_weight = retriever.alpha + retriever.beta + retriever.gamma
    
    # Total weight should be close to 1.0 for normalized scoring
    assert 0.9 <= total_weight <= 1.1, f"Weights should sum to ~1.0, got {total_weight}"


def test_hybrid_retriever_load_graceful_fallback() -> None:
    """Test that retriever gracefully handles missing index files."""
    retriever = HybridRetriever()
    result = retriever.load()
    
    # Should return False if indices don't exist, not raise exception
    assert isinstance(result, bool)
    if not result:
        # If not ready, is_ready should be False
        assert not retriever.is_ready


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
