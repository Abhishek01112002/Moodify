"""
Self-Supervised Item Tower Training
====================================
Strategy: SimCLR-style contrastive learning on audio features.

For each track, two augmented views (Gaussian noise) are created.
NT-Xent loss trains the encoder to embed same-track views close
together, and different-track views far apart — without any user labels.

Result: 64-dim L2-normalized embeddings that capture audio similarity
better than raw z-scored features.

Usage:
    poetry run python src/models/train_item_tower.py

Outputs:
    models/item_tower/         — SavedModel encoder
    src/retrieval/tt_faiss.index
    src/retrieval/tt_metadata.pkl
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pickle
import logging
import numpy as np
import faiss
import mlflow
import tensorflow as tf
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("ItemTower")

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_DIM  = 64       # compact: input is only 9 features
BATCH_SIZE     = 2048
EPOCHS         = 30
LR             = 3e-4
NOISE_STD      = 0.15     # augmentation noise std
TEMPERATURE    = 0.07     # NT-Xent temperature
WARMUP_EPOCHS  = 3

METADATA_PATH  = Path("src/retrieval/metadata.pkl")
ENCODER_PATH   = Path("models/item_tower")
TT_INDEX_PATH  = Path("src/retrieval/tt_faiss.index")
TT_META_PATH   = Path("src/retrieval/tt_metadata.pkl")

FEATURE_COLS = [
    "danceability", "energy", "valence", "tempo",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "popularity",
]


# ── NT-Xent Loss ──────────────────────────────────────────────────────────────
def nt_xent_loss(z1: tf.Tensor, z2: tf.Tensor, temperature: float = 0.07) -> tf.Tensor:
    """
    Normalized Temperature-scaled Cross-Entropy (SimCLR loss).
    z1, z2: [B, D] — L2-normalized embeddings of two augmented views.
    Positive pairs: (z1[i], z2[i])
    Negatives: all other pairs within the batch.
    """
    B = tf.shape(z1)[0]
    z = tf.concat([z1, z2], axis=0)               # [2B, D]
    sim = tf.matmul(z, z, transpose_b=True)        # [2B, 2B] — cosine (L2-normed)
    sim = sim / temperature

    # Mask self-similarity on diagonal
    mask = tf.eye(2 * B, dtype=tf.bool)
    NEG_INF = tf.constant(-1e9, dtype=tf.float32)
    sim = tf.where(mask, NEG_INF, sim)

    # Positive indices: for row i in [0,B) → positive is i+B; for i in [B,2B) → i-B
    pos_upper = tf.range(B, 2 * B, dtype=tf.int32)
    pos_lower = tf.range(0, B,     dtype=tf.int32)
    labels = tf.concat([pos_upper, pos_lower], axis=0)  # [2B]

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=sim
    )
    return tf.reduce_mean(loss)


# ── Item Encoder ──────────────────────────────────────────────────────────────
def build_encoder(input_dim: int, emb_dim: int) -> tf.keras.Model:
    """
    3-layer MLP with BN + dropout, produces L2-normalized embeddings.
    """
    inp = tf.keras.Input(shape=(input_dim,), name="audio_features")
    x   = tf.keras.layers.Dense(256, activation="elu")(inp)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.Dropout(0.15)(x)
    x   = tf.keras.layers.Dense(128, activation="elu")(x)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.Dense(emb_dim, name="embedding")(x)
    out = tf.keras.layers.Lambda(
        lambda v: tf.nn.l2_normalize(v, axis=-1), name="l2_norm"
    )(x)
    return tf.keras.Model(inputs=inp, outputs=out, name="ItemEncoder")


# ── Data Pipeline ─────────────────────────────────────────────────────────────
def build_dataset(features: np.ndarray, noise_std: float, batch_size: int):
    """
    Yields (view1, view2) augmented pairs from the feature matrix.
    view2 = view1 + Gaussian noise  (same track, slightly perturbed).
    """
    n, d = features.shape

    def augment(x):
        n1 = tf.random.normal(tf.shape(x), stddev=noise_std)
        n2 = tf.random.normal(tf.shape(x), stddev=noise_std)
        return x + n1, x + n2

    ds = (
        tf.data.Dataset.from_tensor_slices(features.astype(np.float32))
        .shuffle(min(n, 100_000), reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=True)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds


# ── Cosine LR with Warmup ─────────────────────────────────────────────────────
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_max, warmup_steps, total_steps):
        super().__init__()
        self.lr_max       = lr_max
        self.warmup_steps = float(warmup_steps)
        self.total_steps  = float(total_steps)

    def __call__(self, step):
        step   = tf.cast(step, tf.float32)
        warmup = self.lr_max * (step / self.warmup_steps)
        cosine = 0.5 * self.lr_max * (
            1.0 + tf.cos(
                np.pi * (step - self.warmup_steps)
                / (self.total_steps - self.warmup_steps)
            )
        )
        return tf.where(step < self.warmup_steps, warmup, cosine)

    def get_config(self):
        return dict(lr_max=self.lr_max,
                    warmup_steps=int(self.warmup_steps),
                    total_steps=int(self.total_steps))


# ── Training Loop ─────────────────────────────────────────────────────────────
@tf.function
def train_step(encoder, optimizer, v1, v2, temperature):
    with tf.GradientTape() as tape:
        z1   = encoder(v1, training=True)
        z2   = encoder(v2, training=True)
        loss = nt_xent_loss(z1, z2, temperature)
    grads = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(grads, encoder.trainable_variables))
    return loss


def train(encoder, dataset, epochs, lr, warmup_epochs, steps_per_epoch):
    total_steps  = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    schedule  = WarmupCosineDecay(lr, warmup_steps, total_steps)
    optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)

    loss_history = []
    for epoch in range(1, epochs + 1):
        epoch_losses = []
        for v1, v2 in dataset:
            loss = train_step(encoder, optimizer, v1, v2, TEMPERATURE)
            epoch_losses.append(float(loss))

        avg = np.mean(epoch_losses)
        loss_history.append(avg)
        mlflow.log_metric("loss", avg, step=epoch)
        log.info(f"Epoch {epoch:>3}/{epochs}  loss={avg:.4f}")

    return loss_history


# ── FAISS Index Rebuild ────────────────────────────────────────────────────────
def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Inner-product index on L2-normalized vectors = cosine similarity."""
    d     = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))
    return index


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Load data
    log.info("Loading metadata...")
    with open(METADATA_PATH, "rb") as f:
        meta = pickle.load(f)

    df      = meta["df"]
    present = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(present)
    if missing:
        log.warning(f"Missing features: {missing} — using {present}")

    features = df[present].values.astype(np.float32)
    n, input_dim = features.shape
    log.info(f"Dataset: {n:,} tracks × {input_dim} features")

    steps_per_epoch = n // BATCH_SIZE

    # Build model
    encoder = build_encoder(input_dim, EMBEDDING_DIM)
    encoder.summary()

    # Dataset
    ds = build_dataset(features, NOISE_STD, BATCH_SIZE)

    # Train
    ENCODER_PATH.mkdir(parents=True, exist_ok=True)
    TT_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("item_tower_self_supervised")
    with mlflow.start_run(run_name="nt_xent_v1"):
        mlflow.log_params({
            "embedding_dim":  EMBEDDING_DIM,
            "batch_size":     BATCH_SIZE,
            "epochs":         EPOCHS,
            "lr":             LR,
            "noise_std":      NOISE_STD,
            "temperature":    TEMPERATURE,
            "n_tracks":       n,
            "n_features":     input_dim,
        })

        log.info("Training self-supervised Item Tower (NT-Xent)...")
        loss_hist = train(encoder, ds, EPOCHS, LR, WARMUP_EPOCHS, steps_per_epoch)

        mlflow.log_metric("final_loss", loss_hist[-1])
        log.info(f"Training complete. Final loss: {loss_hist[-1]:.4f}")

        # Save encoder
        tf.saved_model.save(encoder, str(ENCODER_PATH))
        log.info(f"Encoder saved to {ENCODER_PATH}")

        # Inference — embed all tracks
        log.info("Generating embeddings for all tracks...")
        embeddings = encoder.predict(features, batch_size=4096, verbose=1)
        log.info(f"Embeddings shape: {embeddings.shape}")

        # Build FAISS
        log.info("Building FAISS index...")
        index = build_faiss_index(embeddings)
        faiss.write_index(index, str(TT_INDEX_PATH))
        log.info(f"FAISS index saved: {TT_INDEX_PATH} ({index.ntotal:,} vectors)")

        # Save metadata (keep df + embeddings for lookup)
        tt_meta = {
            "df":             df,
            "feature_cols":   present,
            "embeddings":     embeddings,
            "embedding_dim":  EMBEDDING_DIM,
        }
        with open(TT_META_PATH, "wb") as f:
            pickle.dump(tt_meta, f)
        log.info(f"Metadata saved: {TT_META_PATH}")

    log.info("Done! Run the app and set RETRIEVER=hybrid to use Two-Tower embeddings.")


if __name__ == "__main__":
    main()
