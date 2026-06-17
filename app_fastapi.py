# app_fastapi.py
"""
Production-ready FastAPI backend for Moodify.
Serves a high-fidelity modern Single Page Application (SPA) frontend.
"""

import html
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import spotipy
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from spotipy.oauth2 import SpotifyClientCredentials

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
load_dotenv(ROOT / ".env")

from src.retrieval.faiss_retriever import SpotifyFAISSRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.search.smart_search import SmartSearchEngine


# Helper functions defined locally to avoid side-effect imports from Streamlit
def safe_text(value: object) -> str:
    return html.escape(str(value)) if value is not None else ""


def feature_label(value: float, high: str, low: str, neutral: str) -> str:
    if value >= 0.65:
        return high
    if value <= -0.65:
        return low
    return neutral


def explain_row(row: pd.Series, mode: str, engine_name: str) -> list[str]:
    reasons: list[str] = []

    score = row.get("final_score", row.get("_search_score", None))
    if pd.notna(score):
        reasons.append(f"ranked highly by the {engine_name} scorer ({float(score):.3f})")

    similarity = row.get("similarity_score", row.get("sim_score", None))
    if pd.notna(similarity):
        reasons.append(f"strong vector similarity to the seed track ({float(similarity):.3f})")

    if pd.notna(row.get("_search_score", None)) and mode == "vibe":
        reasons.append("matches the requested vibe profile")

    if pd.notna(row.get("pop_score", None)):
        reasons.append(f"popularity signal contributed {float(row['pop_score']):.2f}")
    elif pd.notna(row.get("popularity", None)):
        reasons.append(f"popularity is {float(row['popularity']):.1f}")

    if pd.notna(row.get("diversity_bonus", None)):
        reasons.append(f"artist diversity bonus {float(row['diversity_bonus']):.2f}")

    if pd.notna(row.get("energy", None)):
        reasons.append(feature_label(float(row["energy"]), "high energy", "low energy", "balanced energy"))
    if pd.notna(row.get("valence", None)):
        reasons.append(feature_label(float(row["valence"]), "brighter mood", "more melancholic mood", "balanced mood"))
    if pd.notna(row.get("danceability", None)):
        reasons.append(feature_label(float(row["danceability"]), "dance-forward feel", "less dance-focused feel", "moderate danceability"))

    return reasons[:4]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("moodify_fastapi")

app = FastAPI(
    title="Moodify API",
    description="Backend API for the Spotify Hybrid Music Recommendation Engine",
    version="1.0.0"
)

# Global variables for model resources
sp_client: Optional[spotipy.Spotify] = None
retriever: Optional[SpotifyFAISSRetriever] = None
engine: Optional[SmartSearchEngine] = None
hybrid_retriever: Optional[HybridRetriever] = None


def get_spotify_env(name: str) -> str:
    return os.getenv(name) or os.getenv(name.replace("SPOTIFY_", "SPOTIPY_"), "")


def get_spotify_client() -> Optional[spotipy.Spotify]:
    client_id = get_spotify_env("SPOTIFY_CLIENT_ID")
    client_secret = get_spotify_env("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        logger.warning("Spotify credentials not found; preview features disabled.")
        return None
    try:
        return spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret,
            )
        )
    except Exception as e:
        logger.error(f"Failed to initialize Spotify client: {e}")
        return None


# Startup event to load model resources
@app.on_event("startup")
def startup_event():
    global sp_client, retriever, engine, hybrid_retriever
    sp_client = get_spotify_client()
    
    try:
        retriever = SpotifyFAISSRetriever()
        retriever.load_index()
        
        engine = SmartSearchEngine(
            df=retriever.df,
            feature_columns=retriever.feature_columns,
            faiss_mean=retriever.mean,
            faiss_std=retriever.std,
            faiss_index=retriever.index,
        )
        
        hybrid = HybridRetriever()
        if hybrid.load():
            hybrid_retriever = hybrid
            logger.info("Loaded Two-Tower hybrid retriever successfully.")
        else:
            logger.warning("Two-tower hybrid retriever files not found. Fallbacks active.")
            
        logger.info("Retrieval models and data loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize retrieval models: {e}", exc_info=True)


def search_spotify_track(sp: Optional[spotipy.Spotify], row: pd.Series) -> Optional[dict]:
    if not sp:
        return None
    try:
        query = f"track:{row.get('name', '')}"
        artist = row.get("artists", row.get("artist_names", ""))
        if artist:
            # handle lists or commas
            if isinstance(artist, list):
                artist_str = artist[0]
            elif isinstance(artist, str):
                artist_str = artist.split(",")[0]
            else:
                artist_str = str(artist)
            query += f" artist:{artist_str}"
            
        info = sp.search(q=query, type="track", limit=1)
        return info["tracks"]["items"][0] if info["tracks"]["items"] else None
    except Exception as e:
        logger.debug(f"Spotify search failed for {row.get('name', 'unknown')}: {e}")
        return None


# Helper to convert pandas series results to JSON serializable structures
def serialize_track(row: pd.Series, rank: int, mode: str, engine_name: str) -> Dict[str, Any]:
    global sp_client
    sp_track = search_spotify_track(sp_client, row)
    
    # Extract audio features
    energy = float(row.get("energy", 0.0) or 0.0)
    dance = float(row.get("danceability", 0.0) or 0.0)
    valence = float(row.get("valence", 0.0) or 0.0)
    
    sim_score = row.get("similarity_score", row.get("sim_score", None))
    if pd.isna(sim_score) or sim_score is None:
        sim_score = row.get("final_score", row.get("_search_score", 0.0))
        
    preview_url = ""
    spotify_url = "#"
    if sp_track:
        preview_url = sp_track.get("preview_url") or ""
        spotify_url = sp_track["external_urls"].get("spotify", "#")
        
    reasons = explain_row(row, mode, engine_name)
    
    return {
        "rank": rank,
        "id": str(row.get("id", row.get("track_id", rank))),
        "name": str(row.get("name", "Unknown")),
        "artist": str(row.get("artists", row.get("artist_names", "Unknown"))),
        "popularity": float(row.get("popularity", 0.0) or 0.0),
        "energy": energy,
        "danceability": dance,
        "valence": valence,
        "similarity": float(sim_score),
        "preview_url": preview_url,
        "spotify_url": spotify_url,
        "reasons": reasons
    }


@app.get("/api/stats")
def get_stats():
    """Retrieve database & engine configurations"""
    if retriever is None or retriever.df is None:
        raise HTTPException(status_code=503, detail="Model assets not loaded yet")
    return {
        "total_tracks": len(retriever.df),
        "hybrid_available": hybrid_retriever is not None,
        "engines": 4 if hybrid_retriever is not None else 3
    }


@app.get("/api/search")
def search(
    q: str = Query(..., description="Query terms or vibe description"),
    top_k: int = Query(8, ge=1, le=50),
    use_hybrid: bool = Query(False, description="Whether to run Two-Tower hybrid recommendations for tracks")
):
    """Run Smart Search & Retrieval Pipeline"""
    if engine is None or retriever is None:
        raise HTTPException(status_code=503, detail="Model assets not loaded yet")
        
    started = time.time()
    query = q.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    # Get seed and FAISS results
    seed, results, mode, label = engine.get_seed_and_recommend(query, retriever, top_k=top_k)
    engine_name = "FAISS"
    
    # Apply hybrid two-tower retrieval if requested and available
    if use_hybrid and hybrid_retriever is not None and seed is not None:
        seed_id = seed.get("id")
        if seed_id is not None:
            try:
                results = hybrid_retriever.recommend_by_id(str(seed_id), top_k=top_k)
                engine_name = "two-tower hybrid"
                mode = "hybrid"
            except KeyError:
                engine_name = "FAISS fallback"
                
    elapsed = time.time() - started
    
    # Serialize results
    tracks = []
    if isinstance(results, pd.DataFrame) and not results.empty:
        for idx, (_, row) in enumerate(results.iterrows(), start=1):
            tracks.append(serialize_track(row, idx, mode, engine_name))
            
    seed_data = None
    if seed is not None:
        seed_data = {
            "name": str(seed.get("name", "")),
            "artist": str(seed.get("artists", seed.get("artist_names", ""))),
            "id": str(seed.get("id", ""))
        }
        
    return {
        "query": query,
        "mode": mode,
        "label": label,
        "elapsed": elapsed,
        "engine_name": engine_name,
        "seed": seed_data,
        "tracks": tracks
    }


@app.get("/api/recommendations/mood/{mood}")
def get_mood_recommendations(mood: str):
    """Get top popular songs filtered by a primary mood label"""
    if retriever is None or retriever.df is None:
        raise HTTPException(status_code=503, detail="Model assets not loaded yet")
        
    # Find matching rows in dataframe
    df = retriever.df
    mood_col = "mood_label" if "mood_label" in df.columns else "primary_mood"
    
    if mood_col not in df.columns:
        # Fallback to general popularity if mood label isn't present
        mood_df = df.nlargest(10, 'popularity')
        actual_mood = "Popular Tracks (Mood Column Missing)"
    else:
        # Match case insensitive
        mood_df = df[df[mood_col].str.lower() == mood.lower()]
        if mood_df.empty:
            # Partial match
            mood_df = df[df[mood_col].str.contains(mood, case=False, na=False)]
            
        if mood_df.empty:
            raise HTTPException(status_code=404, detail=f"No tracks found for mood '{mood}'")
            
        mood_df = mood_df.nlargest(10, 'popularity')
        actual_mood = mood.title()
        
    tracks = []
    for idx, (_, row) in enumerate(mood_df.iterrows(), start=1):
        tracks.append(serialize_track(row, idx, "vibe", "Mood Popularity"))
        
    return {
        "mood": actual_mood,
        "tracks": tracks
    }


# Static files and SPA serving
# Ensure paths exist
Path("app/templates").mkdir(parents=True, exist_ok=True)
Path("app/static").mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def read_root():
    """Serve SPA index.html"""
    index_path = Path("app/templates/index.html")
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_path)


if __name__ == "__main__":
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=8000, reload=True)
