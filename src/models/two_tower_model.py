"""Profile-aware two-tower recommender training.

The older version of this script used random user vectors. This version builds
real user/profile vectors from the data itself. If playlist IDs are available,
each playlist becomes a profile. Otherwise the script falls back to mood labels
or genre labels.

For each track, the user/profile tower receives the average feature vector of
the other tracks in the same profile. The item tower receives the track's own
feature vector. This creates a defensible playlist-to-track retrieval training
task without inventing synthetic random users.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
from sklearn.preprocessing import StandardScaler


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("ProfileTwoTower")


DEFAULT_DATA_PATHS = [
    Path("data/processed/tracks_large_cleaned.parquet"),
    Path("data/processed/tracks_cleaned.parquet"),
    Path("spotify_songs_cleaned.csv"),
]

PROFILE_COLUMNS = [
    "playlist_id",
    "playlist_name",
    "mood_label",
    "primary_genre",
    "genres",
    "track_genre",
]

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "popularity",
]


@dataclass
class TrainingArtifacts:
    rows: int
    profile_column: str
    profile_count: int
    feature_count: int
    embedding_dim: int
    epochs: int
    batch_size: int
    final_loss: float
    model_dir: str
    item_tower_dir: str
    user_tower_dir: str
    faiss_index_path: str
    metadata_path: str


def resolve_data_path(path: Path | None) -> Path:
    if path:
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        return path

    for candidate in DEFAULT_DATA_PATHS:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No processed dataset found. Expected one of: "
        + ", ".join(str(p) for p in DEFAULT_DATA_PATHS)
    )


def load_frame(path: Path) -> pd.DataFrame:
    log.info("Loading training data from %s", path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "id" not in df.columns and "track_id" in df.columns:
        df = df.rename(columns={"track_id": "id"})
    if "artists" not in df.columns and "artist_names" in df.columns:
        df = df.rename(columns={"artist_names": "artists"})

    if "id" not in df.columns:
        df["id"] = np.arange(len(df)).astype(str)
    if "name" not in df.columns:
        df["name"] = df["id"].astype(str)

    return df


def choose_profile_column(df: pd.DataFrame) -> str:
    for column in PROFILE_COLUMNS:
        if column in df.columns and df[column].notna().nunique() > 1:
            return column
    raise ValueError(
        "No usable profile column found. Add playlist_id, mood_label, "
        "primary_genre, genres, or track_genre."
    )


def choose_feature_columns(df: pd.DataFrame) -> list[str]:
    genre_features = [column for column in df.columns if column.startswith("genre_")]
    candidates = [column for column in AUDIO_FEATURES if column in df.columns]
    candidates.extend(genre_features)

    for optional in ["artist_popularity", "artist_followers", "release_year", "duration_ms"]:
        if optional in df.columns:
            candidates.append(optional)

    feature_columns = [
        column
        for column in dict.fromkeys(candidates)
        if pd.api.types.is_numeric_dtype(df[column])
    ]

    if not feature_columns:
        ignored = set(PROFILE_COLUMNS + ["id"])
        feature_columns = [
            column
            for column in df.select_dtypes(include=[np.number]).columns
            if column not in ignored
        ]

    if not feature_columns:
        raise ValueError("No numeric feature columns available for two-tower training.")

    return feature_columns


def build_profile_examples(
    df: pd.DataFrame,
    profile_column: str,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, StandardScaler]:
    work = df.dropna(subset=[profile_column]).reset_index(drop=True)
    work[profile_column] = work[profile_column].astype(str)

    raw_features = (
        work[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )

    scaler = StandardScaler()
    item_features = scaler.fit_transform(raw_features).astype(np.float32)

    feature_frame = pd.DataFrame(item_features, columns=feature_columns)
    group_sizes = work[profile_column].map(work[profile_column].value_counts()).to_numpy()
    group_sums = feature_frame.groupby(work[profile_column]).transform("sum").to_numpy()

    counts = group_sizes.reshape(-1, 1).astype(np.float32)
    user_features = np.where(
        counts > 1,
        (group_sums - item_features) / np.maximum(counts - 1, 1),
        group_sums / np.maximum(counts, 1),
    ).astype(np.float32)

    return work, user_features, item_features, scaler


def build_tower(input_dim: int, embedding_dim: int, name: str) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name=f"{name}_features")
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(embedding_dim)(x)
    outputs = tf.keras.layers.Lambda(lambda v: tf.nn.l2_normalize(v, axis=-1))(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


class ProfileTwoTower(tfrs.Model):
    def __init__(self, user_tower: tf.keras.Model, item_tower: tf.keras.Model):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.task = tfrs.tasks.Retrieval()

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_tower(features["user_features"], training=training)
        item_embeddings = self.item_tower(features["item_features"], training=training)
        return self.task(user_embeddings, item_embeddings)


def build_dataset(
    user_features: np.ndarray,
    item_features: np.ndarray,
    batch_size: int,
) -> tf.data.Dataset:
    return (
        tf.data.Dataset.from_tensor_slices(
            {
                "user_features": user_features,
                "item_features": item_features,
            }
        )
        .shuffle(min(len(user_features), 100_000), reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def train(args: argparse.Namespace) -> TrainingArtifacts:
    data_path = resolve_data_path(args.data)
    df = normalize_schema(load_frame(data_path))
    profile_column = args.profile_column or choose_profile_column(df)
    feature_columns = choose_feature_columns(df)

    work, user_features, item_features, scaler = build_profile_examples(
        df, profile_column, feature_columns
    )

    log.info(
        "Training on %s rows, %s profiles, %s features. Profile column: %s",
        len(work),
        work[profile_column].nunique(),
        len(feature_columns),
        profile_column,
    )

    dataset = build_dataset(user_features, item_features, args.batch_size)
    user_tower = build_tower(user_features.shape[1], args.embedding_dim, "user_tower")
    item_tower = build_tower(item_features.shape[1], args.embedding_dim, "item_tower")
    model = ProfileTwoTower(user_tower, item_tower)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate))

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.index_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("profile_two_tower")
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(
            {
                "data_path": str(data_path),
                "profile_column": profile_column,
                "profile_count": work[profile_column].nunique(),
                "feature_count": len(feature_columns),
                "embedding_dim": args.embedding_dim,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
            }
        )

        history = model.fit(dataset, epochs=args.epochs, verbose=1)
        final_loss = float(history.history["loss"][-1])
        mlflow.log_metric("final_loss", final_loss)

        tf.saved_model.save(model, str(args.model_dir))
        tf.saved_model.save(user_tower, str(args.user_tower_dir))
        tf.saved_model.save(item_tower, str(args.item_tower_dir))

        item_embeddings = item_tower.predict(item_features, batch_size=4096, verbose=1)
        index = faiss.IndexFlatIP(item_embeddings.shape[1])
        index.add(item_embeddings.astype(np.float32))
        faiss.write_index(index, str(args.index_path))

        metadata = {
            "df": work.reset_index(drop=True),
            "profile_column": profile_column,
            "feature_columns": feature_columns,
            "feature_mean": scaler.mean_,
            "feature_scale": scaler.scale_,
            "item_embeddings": item_embeddings,
            "embedding_dim": args.embedding_dim,
        }
        with open(args.metadata_path, "wb") as f:
            pickle.dump(metadata, f)

    artifacts = TrainingArtifacts(
        rows=len(work),
        profile_column=profile_column,
        profile_count=work[profile_column].nunique(),
        feature_count=len(feature_columns),
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        final_loss=final_loss,
        model_dir=str(args.model_dir),
        item_tower_dir=str(args.item_tower_dir),
        user_tower_dir=str(args.user_tower_dir),
        faiss_index_path=str(args.index_path),
        metadata_path=str(args.metadata_path),
    )

    log.info("Training complete:\n%s", json.dumps(asdict(artifacts), indent=2))
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a profile-aware two-tower model.")
    parser.add_argument("--data", type=Path, default=None, help="CSV or Parquet dataset.")
    parser.add_argument(
        "--profile-column",
        default=None,
        help="Column to use as user/profile ID. Defaults to playlist, mood, or genre.",
    )
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--run-name", default="profile_two_tower_v1")
    parser.add_argument("--model-dir", type=Path, default=Path("models/profile_two_tower"))
    parser.add_argument("--user-tower-dir", type=Path, default=Path("models/profile_user_tower"))
    parser.add_argument("--item-tower-dir", type=Path, default=Path("models/profile_item_tower"))
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("src/retrieval/profile_tt_faiss.index"),
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("src/retrieval/profile_tt_metadata.pkl"),
    )
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
