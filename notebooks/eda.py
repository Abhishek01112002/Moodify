import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

print("🚀 EDA Script Starting...\n")

# Load cleaned data
df = pd.read_parquet("data/processed/tracks_cleaned.parquet")
print(f"✅ Loaded {df.shape[0]:,} tracks with {df.shape[1]} columns\n")
print("Columns:", df.columns.tolist())

# 1. Basic Stats
print("\n📊 Basic Statistics:")
print(df.describe().round(3))

# 2. Top Genres & Artists
if "genres" in df.columns:
    print("\n🎵 Top 10 Genres:")
    print(df["genres"].value_counts().head(10))

if "track_artists" in df.columns:
    print("\n🎤 Top 10 Artists:")
    print(df["track_artists"].value_counts().head(10))

# 3. Most Popular Tracks
if "popularity" in df.columns and "name" in df.columns:
    print("\n🔥 Top 10 Most Popular Tracks:")
    desired_cols = ["name", "track_artists", "popularity", "danceability", "energy", "valence"]
    existing_cols = [c for c in desired_cols if c in df.columns]
    top10 = df.nlargest(10, "popularity")[existing_cols]
    print(top10)

# 4. Audio Features Distribution (Plots save ho jayenge)
audio_cols = ["danceability", "energy", "valence", "tempo", "speechiness", 
              "acousticness", "instrumentalness", "liveness", "popularity"]

for col in audio_cols:
    if col in df.columns:
        fig = px.histogram(df, x=col, nbins=80, title=f"Distribution of {col.title()}", 
                          color_discrete_sequence=["#1DB954"])
        fig.write_image(f"eda_{col}.png")           # plot save ho jaayega
        print(f"   Saved plot: eda_{col}.png")

# 5. Correlation Heatmap
numeric_cols = df.select_dtypes(include="number").columns.tolist()
plt.figure(figsize=(12, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Correlation between Audio Features")
plt.tight_layout()
plt.savefig("eda_correlation_heatmap.png", dpi=300)
print("   Saved: eda_correlation_heatmap.png")

print("\n🎉 EDA Complete! Sab plots 'eda_*.png' naam se save ho gaye hain.")
print("Ab mujhe batao output mein kya interesting dikha (top genres, top songs, etc.)")
