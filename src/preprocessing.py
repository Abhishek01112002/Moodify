import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("🚀 Starting preprocessing for large Spotify dataset...")

# Adjust filename according to your downloaded file
input_file = RAW_DIR / "dataset.csv" / "dataset.csv"   # ← yahan apna exact filename daal do

df = pd.read_csv(input_file)

print(f"✅ Loaded {df.shape[0]:,} tracks with {df.shape[1]} columns")

# Rename columns to match expected schema if they use track_ prefix
df = df.rename(columns={
    'track_id': 'id',
    'track_name': 'name',
    'track_genre': 'genres'
})

# Important columns (audio features + metadata)
key_columns = [
    'id', 'name', 'artists', 'track_artists', 'genres', 'popularity',
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_ms'
]

# Jo columns actually available hain unko lo
available_cols = [col for col in key_columns if col in df.columns]
df = df[available_cols].copy()

# Basic cleaning
df = df.dropna(subset=['id', 'name']).drop_duplicates(subset=['id'])

# Normalize numerical audio features
audio_features = [col for col in ['danceability', 'energy', 'valence', 'tempo', 
                                  'speechiness', 'acousticness', 'instrumentalness', 
                                  'liveness', 'popularity'] if col in df.columns]

for col in audio_features:
    mean = df[col].mean()
    std = df[col].std() + 1e-8
    df[col] = (df[col] - mean) / std

print(f"✅ After cleaning: {df.shape[0]:,} tracks")
print(f"Available audio features: {audio_features}")

# Save as Parquet (fast & compressed)
output_file = PROCESSED_DIR / "tracks_large_cleaned.parquet"
df.to_parquet(output_file, compression="zstd", index=False)

print(f"🎉 Preprocessing complete! Saved to {output_file}")
print(f"File size: {output_file.stat().st_size / (1024*1024):.1f} MB")
