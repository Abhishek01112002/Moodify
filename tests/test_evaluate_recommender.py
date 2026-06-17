import pandas as pd

from scripts.evaluate_recommender import evaluate


def test_evaluate_returns_metrics_for_proxy_labels():
    df = pd.DataFrame(
        {
            "track_id": ["a", "b", "c", "d", "e", "f"],
            "name": ["A", "B", "C", "D", "E", "F"],
            "artist_names": ["one", "two", "three", "four", "five", "six"],
            "mood_label": ["chill", "chill", "party", "party", "focus", "focus"],
            "danceability": [0.2, 0.25, 0.9, 0.85, 0.1, 0.15],
            "energy": [0.3, 0.35, 0.95, 0.9, 0.2, 0.25],
            "valence": [0.4, 0.45, 0.8, 0.75, 0.3, 0.35],
            "popularity": [20, 25, 80, 75, 30, 35],
        }
    )

    result = evaluate(df, top_k=1, sample_size=6, seed=7)

    assert result.rows == 6
    assert result.label_column == "mood_label"
    assert result.feature_count >= 4
    assert 0.0 <= result.precision_at_k <= 1.0
    assert 0.0 <= result.ndcg_at_k <= 1.0
    assert result.avg_latency_ms >= 0.0
