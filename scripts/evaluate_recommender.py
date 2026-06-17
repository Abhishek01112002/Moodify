import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class EvaluationResult:
    """Container for offline recommendation evaluation metrics."""

    rows: int
    label_column: str
    feature_count: int
    precision_at_k: float
    recall_at_k: float
    map_at_k: float
    ndcg_at_k: float
    catalog_coverage: float
    artist_diversity: float
    avg_latency_ms: float


# Columns that should never be treated as audio features
_IDENTITY_COLS = {
    "id",
    "track_id",
    "name",
    "artists",
    "artist_names",
    "album",
    "release_date",
    "primary_genre",
    "all_genres",
    "mood",
    "mood_label",
    "playlist_id",
    "playlist_name",
    "genres",
    "track_genre",
    "artist_id",
    "artist_ids",
    "artist_followers",
    "artist_popularity",
}


def _choose_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric columns that are not identity metadata."""
    return [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in _IDENTITY_COLS
    ]


def evaluate(
    df: pd.DataFrame,
    top_k: int = 10,
    sample_size: int = 200,
    seed: int = 42,
    label_column: str = "mood_label",
) -> EvaluationResult:
    """Offline evaluation using cosine similarity over audio features.

    Parameters
    ----------
    df : pd.DataFrame
        Track dataset with numeric audio features and a ``label_column``.
    top_k : int
        Number of recommendations to evaluate per seed track.
    sample_size : int
        Number of seed tracks to sample from ``df``.
    seed : int
        Random seed for reproducibility.
    label_column : str
        Column used as proxy relevance label (e.g. ``mood_label`` or ``track_genre``).

    Returns
    -------
    EvaluationResult
    """
    df = df.copy().reset_index(drop=True)
    n_rows = len(df)

    feature_cols = _choose_feature_columns(df)
    if not feature_cols:
        raise ValueError("No numeric feature columns found for evaluation.")

    # Sample seeds
    sample_size = min(sample_size, n_rows)
    rng = np.random.default_rng(seed)
    seed_indices = rng.choice(n_rows, size=sample_size, replace=False)

    # Normalised feature matrix
    features = df[feature_cols].fillna(0).values.astype(np.float32)
    means = features.mean(axis=0)
    stds = features.std(axis=0) + 1e-8
    features_norm = (features - means) / stds

    # Accumulators
    all_recommended: set[int] = set()
    total_precision = 0.0
    total_recall = 0.0
    total_ap = 0.0
    total_ndcg = 0.0
    total_latency = 0.0
    total_artist_diversity = 0.0
    queries = 0

    for idx in seed_indices:
        seed_label = df.iloc[idx].get(label_column)
        if pd.isna(seed_label):
            continue

        start = time.perf_counter()

        # Cosine similarity against all tracks
        query_vec = features_norm[idx : idx + 1]
        sims = cosine_similarity(query_vec, features_norm).flatten()
        sims[idx] = -1.0  # exclude self

        top_indices = np.argsort(sims)[::-1][:top_k]
        elapsed = (time.perf_counter() - start) * 1000.0
        total_latency += elapsed

        # Relevance check: same label as seed
        rec_labels = df.iloc[top_indices][label_column].values
        relevant = np.array([str(l) == str(seed_label) for l in rec_labels])

        # Precision@K
        precision = relevant.sum() / top_k
        total_precision += precision

        # Recall@K
        total_relevant = (df[label_column] == seed_label).sum()
        recall = relevant.sum() / max(total_relevant - 1, 1)
        total_recall += recall

        # MAP@K
        ap = 0.0
        hits = 0
        for rank, rel in enumerate(relevant, start=1):
            if rel:
                hits += 1
                ap += hits / rank
        ap /= max(top_k, 1)
        total_ap += ap

        # NDCG@K
        dcg = sum(
            (1.0 / np.log2(rank + 1))
            for rank, rel in enumerate(relevant, start=1)
            if rel
        )
        ideal_rels = min(total_relevant - 1, top_k)
        idcg = sum(
            (1.0 / np.log2(rank + 1)) for rank in range(1, ideal_rels + 1)
        )
        ndcg = dcg / idcg if idcg > 0 else 0.0
        total_ndcg += ndcg

        # Catalog coverage
        all_recommended.update(top_indices.tolist())

        # Artist diversity
        artist_col = "artists" if "artists" in df.columns else "artist_names"
        if artist_col in df.columns:
            artists = df.iloc[top_indices][artist_col].astype(str).unique()
            total_artist_diversity += len(artists) / top_k
        else:
            total_artist_diversity += 1.0

        queries += 1

    if queries == 0:
        return EvaluationResult(
            rows=n_rows,
            label_column=label_column,
            feature_count=len(feature_cols),
            precision_at_k=0.0,
            recall_at_k=0.0,
            map_at_k=0.0,
            ndcg_at_k=0.0,
            catalog_coverage=0.0,
            artist_diversity=0.0,
            avg_latency_ms=0.0,
        )

    return EvaluationResult(
        rows=n_rows,
        label_column=label_column,
        feature_count=len(feature_cols),
        precision_at_k=total_precision / queries,
        recall_at_k=total_recall / queries,
        map_at_k=total_ap / queries,
        ndcg_at_k=total_ndcg / queries,
        catalog_coverage=len(all_recommended) / n_rows,
        artist_diversity=total_artist_diversity / queries,
        avg_latency_ms=total_latency / queries,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline recommendation evaluation harness for Moodify."
    )
    parser.add_argument("--data", required=True, help="Path to CSV or Parquet dataset.")
    parser.add_argument("--top-k", type=int, default=10, help="Cut-off rank for metrics.")
    parser.add_argument(
        "--sample-size", type=int, default=200, help="Number of seed tracks to sample."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--label-column",
        default="mood_label",
        help="Proxy relevance column (e.g. mood_label, track_genre).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file path to write results.",
    )
    args = parser.parse_args()

    path = Path(args.data)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    result = evaluate(
        df,
        top_k=args.top_k,
        sample_size=args.sample_size,
        seed=args.seed,
        label_column=args.label_column,
    )

    print(f"Evaluated {result.rows} rows, {result.feature_count} features")
    print(f"  Precision@{args.top_k}: {result.precision_at_k:.3f}")
    print(f"  Recall@{args.top_k}:    {result.recall_at_k:.3f}")
    print(f"  MAP@{args.top_k}:       {result.map_at_k:.3f}")
    print(f"  NDCG@{args.top_k}:      {result.ndcg_at_k:.3f}")
    print(f"  Catalog coverage:     {result.catalog_coverage:.3f}")
    print(f"  Artist diversity:     {result.artist_diversity:.3f}")
    print(f"  Avg latency:          {result.avg_latency_ms:.2f} ms")

    if args.output:
        out = {
            "rows": result.rows,
            "label_column": result.label_column,
            "feature_count": result.feature_count,
            f"precision@{args.top_k}": result.precision_at_k,
            f"recall@{args.top_k}": result.recall_at_k,
            f"map@{args.top_k}": result.map_at_k,
            f"ndcg@{args.top_k}": result.ndcg_at_k,
            "catalog_coverage": result.catalog_coverage,
            "artist_diversity": result.artist_diversity,
            "avg_latency_ms": result.avg_latency_ms,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
