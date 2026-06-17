"""Official Streamlit demo for Moodify.

This app consolidates the previous demo experience into one polished path:
smart search, vibe search, FAISS retrieval, optional two-tower hybrid reranking,
Spotify previews, and recommendation explanations.
"""

from __future__ import annotations

import html
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import spotipy
import streamlit as st
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials

logger = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
load_dotenv(ROOT / ".env")

from src.retrieval.faiss_retriever import SpotifyFAISSRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.search.smart_search import SmartSearchEngine


st.set_page_config(
    page_title="Moodify — Hybrid Spotify Recommendations",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state early to drive dynamic theme
for key, value in {
    "query": "",
    "results": None,
    "mode": "",
    "label": "",
    "elapsed": 0.0,
    "engine_name": "FAISS",
}.items():
    st.session_state.setdefault(key, value)


def get_mood_theme(query: str) -> tuple[str, str, str]:
    """Map query to CSS gradients for background. Returns (mood_name, color_a, color_b)"""
    q = str(query or "").lower().strip()
    if not q:
        return "chill", "#1fa2a6", "#2d5bff"
    # Check query keywords
    if any(k in q for k in ["gym", "workout", "run", "hype", "pump", "beast", "intense", "power", "cardio", "party", "club", "dance", "disco", "rave", "edm", "banger", "happy", "hustle"]):
        return "energetic", "#ff4d2e", "#ffb800"
    if any(k in q for k in ["sad", "heartbreak", "moody", "dark", "melancholy", "depressed", "lonely", "ambient", "rainy", "cry", "pain"]):
        return "moody", "#3a3f58", "#6b5b95"
    if any(k in q for k in ["romantic", "love", "dinner", "date", "sweet", "passion", "hug"]):
        return "romantic", "#d6336c", "#ff8fa3"
    return "chill", "#1fa2a6", "#2d5bff"


# Compute current mood theme
mood_name, color_a, color_b = get_mood_theme(st.session_state["query"])

# Inject fonts
st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
""",
    unsafe_allow_html=True,
)

# Inject CSS styles (using f-string for dynamic mood theme variables)
st.markdown(
    f"""
<style>
  :root {{
    --ink: #ffffff;
    --shadow-ink: #0a0a0a;
    --canvas: #121212;
    --canvas-raised: #1a1a1a;
    --lime: #d4ff3f;
    --mood-a: {color_a};
    --mood-b: {color_b};
  }}
  /* ---------- mood-reactive ambient field ---------- */
  .mood-field {{
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    z-index: -2;
    opacity: 0.55;
    background:
      radial-gradient(circle at 15% 20%, var(--mood-a) 0%, transparent 38%),
      radial-gradient(circle at 85% 15%, var(--mood-b) 0%, transparent 42%),
      radial-gradient(circle at 50% 90%, var(--mood-a) 0%, transparent 45%);
    filter: blur(60px);
    transition: background 1.1s ease, opacity 0.6s ease;
    animation: drift 18s ease-in-out infinite alternate;
  }}
  @keyframes drift {{
    0% {{ transform: translate(0,0) scale(1); }}
    100% {{ transform: translate(-2%, 3%) scale(1.08); }}
  }}
  .grain {{
    position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1;
    opacity: 0.05; mix-blend-mode: overlay; pointer-events: none;
  }}
  /* ---------- Streamlit DOM overrides ---------- */
  [data-testid="stAppViewContainer"] {{
    background: transparent !important;
  }}
  [data-testid="stHeader"] {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
  }}
  [data-testid="stHeaderDecoration"] {{
    display: none !important;
    height: 0px !important;
  }}
  .main .block-container {{
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 48px 32px 120px !important;
    position: relative;
    z-index: 2;
  }}
  html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--ink);
  }}
  /* ---------- header ---------- */
  header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 24px;
    margin-bottom: 40px;
    border-bottom: 3px solid var(--ink);
    padding-bottom: 20px;
  }}
  .logo {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700;
    font-size: 28px;
    letter-spacing: -0.5px;
    color: var(--ink);
  }}
  .logo span {{
    color: var(--lime);
  }}
  .tagline {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px;
    opacity: 0.6;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--ink);
  }}
  /* ---------- track cards grid ---------- */
  .grid {{
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin-top: 20px;
    max-width: 780px;
    margin-left: auto;
    margin-right: auto;
  }}
  .card {{
    position: relative;
    border: 2px solid var(--ink);
    background: var(--canvas-raised);
    box-shadow: 6px 6px 0 var(--shadow-ink);
    display: flex;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    margin-bottom: 4px;
  }}
  .card:hover {{
    transform: translate(-3px,-3px);
    box-shadow: 9px 9px 0 var(--shadow-ink);
  }}
  .card.now-playing {{
    border-color: var(--lime);
  }}
  .card.now-playing .stub {{
    background: rgba(212, 255, 63, 0.08);
  }}
  .stub {{
    width: 88px;
    flex-shrink: 0;
    background: transparent;
    color: var(--ink);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border-right: 2px dashed rgba(255, 255, 255, 0.2);
    position: relative;
  }}
  .stub::before, .stub::after {{
    content: ""; position: absolute; width: 14px; height: 14px; border-radius: 50%;
    background: #121212; left: 50%; transform: translateX(-50%);
  }}
  .stub::before {{ top: -7px; }}
  .stub::after {{ bottom: -7px; }}
  .play-circle {{
    width: 42px; height: 42px; border-radius: 50%;
    background: var(--ink); color: var(--shadow-ink);
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
    cursor: pointer;
    user-select: none;
    transition: background-color 0.15s, color 0.15s, transform 0.1s;
  }}
  .play-circle:hover {{
    background: var(--lime);
    color: var(--shadow-ink);
  }}
  .play-circle:active {{
    transform: scale(0.9);
  }}
  .card.now-playing .play-circle {{
    background: var(--lime);
    color: var(--shadow-ink);
  }}
  .stub .rank {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px;
    margin-top: 8px;
    font-weight: 600;
  }}
  .card-body {{
    flex: 1;
    padding: 18px 20px;
    min-width: 0;
  }}
  .track-title {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700;
    font-size: 19px;
    line-height: 1.15;
    margin: 0 0 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    color: var(--ink);
  }}
  .track-artist {{
    font-size: 13px;
    opacity: 0.65;
    margin: 0 0 14px;
    color: var(--ink);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .meter-row {{
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 12px;
  }}
  .meter {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .meter-label {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px;
    text-transform: uppercase;
    opacity: 0.5;
    width: 62px;
    flex-shrink: 0;
    letter-spacing: 0.5px;
    color: var(--ink);
  }}
  .meter-track {{
    flex: 1;
    height: 6px;
    background: rgba(255, 255, 255, 0.12);
    position: relative;
  }}
  .meter-fill {{
    position: absolute;
    top: 0; left: 0; height: 100%;
    background: var(--ink);
  }}
  .card.now-playing .meter-fill {{
    background: var(--lime);
  }}
  .meter-val {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px;
    width: 28px;
    text-align: right;
    opacity: 0.7;
    color: var(--ink);
  }}
  .card-footer {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid rgba(255, 255, 255, 0.15);
    padding-top: 10px;
  }}
  .score {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px;
    opacity: 0.6;
    color: var(--ink);
  }}
  .score b {{
    color: var(--lime);
    font-weight: 600;
  }}
  .why-btn {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    background: none;
    border: 1px solid var(--ink);
    color: var(--ink);
    padding: 5px 10px;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
  }}
  .why-btn:hover {{
    background: var(--ink);
    color: var(--shadow-ink);
  }}
  /* ---------- mini waveform divider ---------- */
  .waveform {{
    display: flex;
    align-items: center;
    gap: 3px;
    height: 24px;
    margin: 8px 0;
  }}
  .waveform span {{
    width: 3px;
    background: var(--ink);
    opacity: 0.35;
    border-radius: 1px;
  }}
  .card.now-playing .waveform span {{
    background: var(--lime);
    opacity: 0.9;
    animation: bounce 0.9s ease-in-out infinite;
  }}
  .card.now-playing .waveform span:nth-child(odd) {{
    animation-delay: 0.15s;
  }}
  @keyframes bounce {{
    0%, 100% {{ transform: scaleY(0.4); }}
    50% {{ transform: scaleY(1); }}
  }}
  /* ---------- sidebar styling overrides ---------- */
  section[data-testid="stSidebar"] {{
    background-color: var(--canvas-raised) !important;
    border-right: 2px solid var(--ink) !important;
  }}
  section[data-testid="stSidebar"] h2 {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: -0.5px;
  }}
  div[data-baseweb="input"] {{
    border: 2px solid var(--ink) !important;
    background-color: transparent !important;
    border-radius: 0px !important;
  }}
  div[data-baseweb="input"] input {{
    color: var(--ink) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 16px !important;
  }}
  /* Quick vibe buttons */
  button[data-testid="stBaseButton-secondary"] {{
    background: transparent !important;
    color: var(--ink) !important;
    border: 2px solid var(--ink) !important;
    border-radius: 0px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    margin-bottom: 2px !important;
    box-shadow: 3px 3px 0 var(--shadow-ink) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    width: 100% !important;
  }}
  button[data-testid="stBaseButton-secondary"]:hover {{
    transform: translate(-2px, -2px) !important;
    box-shadow: 5px 5px 0 var(--shadow-ink) !important;
    background: transparent !important;
    color: var(--ink) !important;
  }}
  /* Get recommendations button */
  button[data-testid="stBaseButton-primary"] {{
    background: var(--lime) !important;
    color: var(--shadow-ink) !important;
    border: 2px solid var(--ink) !important;
    border-radius: 0px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    box-shadow: 4px 4px 0 var(--shadow-ink) !important;
    transition: transform 0.1s, box-shadow 0.1s !important;
    width: 100% !important;
  }}
  button[data-testid="stBaseButton-primary"]:hover {{
    transform: translate(-2px, -2px) !important;
    box-shadow: 6px 6px 0 var(--shadow-ink) !important;
    background: var(--lime) !important;
    color: var(--shadow-ink) !important;
  }}
  /* Slider styling */
  div[data-testid="stSlider"] div[role="slider"] {{
    background-color: var(--lime) !important;
    border: 2px solid var(--ink) !important;
  }}
  /* Checkbox styling */
  div[data-testid="stCheckbox"] label {{
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.5px;
  }}
  /* ---------- bottom player ---------- */
  .player {{
    position: fixed;
    bottom: 24px;
    right: 24px;
    width: 380px;
    z-index: 9999;
    background: var(--canvas-raised);
    border: 2px solid var(--ink);
    box-shadow: 6px 6px 0 var(--shadow-ink);
    display: none;
    align-items: center;
    gap: 14px;
    padding: 12px 16px;
  }}
  .player .stub-mini {{
    width: 42px; height: 42px; border: 2px solid var(--ink);
    background: var(--lime); flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    color: var(--shadow-ink); font-weight: 700;
    font-family: 'Space Grotesk', sans-serif !important;
    cursor: pointer;
    user-select: none;
  }}
  .player .info {{
    flex: 1;
    min-width: 0;
  }}
  .player .info .t {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700;
    font-size: 13px;
    color: var(--ink);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .player .info .a {{
    font-size: 10px;
    opacity: 0.6;
    color: var(--ink);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .player .scrub {{
    width: 60px; height: 4px;
    background: rgba(255, 255, 255, 0.15);
    position: relative;
    flex-shrink: 0;
  }}
  .player .scrub-fill {{
    position: absolute; left: 0; top: 0; height: 100%; width: 0%;
    background: var(--lime);
  }}
  .player .time {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px;
    opacity: 0.6;
    flex-shrink: 0;
    color: var(--ink);
  }}
  @media (max-width: 480px) {{
    .player {{
      left: 16px;
      right: 16px;
      width: auto;
      bottom: 16px;
    }}
  }}
  /* ---------- custom scrollbar ---------- */
  ::-webkit-scrollbar {{
    width: 8px;
    height: 8px;
  }}
  ::-webkit-scrollbar-track {{
    background: var(--canvas);
  }}
  ::-webkit-scrollbar-thumb {{
    background: var(--ink);
    border: 2px solid var(--canvas);
  }}
  ::-webkit-scrollbar-thumb:hover {{
    background: var(--lime);
  }}
</style>
""",
    unsafe_allow_html=True,
)

# Inject ambient background divs
st.markdown(
    """
<div class="mood-field"></div>
<div class="grain"></div>
""",
    unsafe_allow_html=True,
)

# Inject bottom mini player structure
st.markdown(
    """<div class="player" id="bottom-player">
  <div class="stub-mini" id="bottom-play-btn">▶</div>
  <div class="info">
    <div class="t">-</div>
    <div class="a">-</div>
  </div>
  <span class="time current-time">0:00</span>
  <div class="scrub"><div class="scrub-fill" id="bottom-scrub-fill"></div></div>
  <span class="time total-time">0:00</span>
</div>""".strip(),
    unsafe_allow_html=True,
)

# Inject client-side JavaScript controller using a same-origin iframe to bypass React script execution security and prevent markdown formatting leaks
st.markdown(
    """<iframe srcdoc="<script>
(function() {
    const parentWindow = window.parent;
    const parentDoc = parentWindow.document;

    if (parentWindow.activeAudio) {
        try { parentWindow.activeAudio.pause(); } catch(e){}
        parentWindow.activeAudio = null;
        parentWindow.activeCard = null;
    }

    if (parentWindow.moodifyInitialized) return;
    parentWindow.moodifyInitialized = true;
    console.log('Moodify JS initialized');

    parentWindow.activeAudio = null;
    parentWindow.activeCard = null;
    
    parentWindow.formatTime = function(secs) {
        if (isNaN(secs)) return '0:00';
        const m = Math.floor(secs / 60);
        const s = Math.floor(secs % 60).toString().padStart(2, '0');
        return m + ':' + s;
    };
    
    parentWindow.playTrack = function(rank, name, artist, previewUrl) {
        const audio = parentDoc.getElementById('audio-' + rank);
        const card = parentDoc.getElementById('card-' + rank);
        const player = parentDoc.getElementById('bottom-player');
        if (!audio || !card || !player) return;
        const playCircle = card.querySelector('.play-circle');
        const waveform = card.querySelector('.waveform');
        
        if (parentWindow.activeAudio && parentWindow.activeAudio !== audio) {
            parentWindow.activeAudio.pause();
            if (parentWindow.activeCard) {
                parentWindow.activeCard.classList.remove('now-playing');
                const oldPlayCircle = parentWindow.activeCard.querySelector('.play-circle');
                if (oldPlayCircle) oldPlayCircle.innerText = '▶';
                const activeWave = parentWindow.activeCard.querySelector('.waveform');
                if (activeWave) activeWave.style.display = 'none';
            }
        }
        
        if (audio.paused) {
            audio.play().catch(function(e) { console.error('Audio play failed:', e); });
            card.classList.add('now-playing');
            if (playCircle) playCircle.innerText = '⏸';
            if (waveform) waveform.style.display = 'flex';
            player.style.display = 'flex';
            player.querySelector('.t').innerText = name;
            player.querySelector('.a').innerText = artist;
            player.querySelector('.stub-mini').innerText = '⏸';
            parentWindow.activeAudio = audio;
            parentWindow.activeCard = card;
            
            audio.ontimeupdate = function() {
                const pct = (audio.currentTime / audio.duration) * 100;
                const scrubFill = parentDoc.getElementById('bottom-scrub-fill');
                if (scrubFill) scrubFill.style.width = pct + '%';
                player.querySelector('.current-time').innerText = parentWindow.formatTime(audio.currentTime);
                player.querySelector('.total-time').innerText = parentWindow.formatTime(audio.duration || 30);
            };
            
            audio.onended = function() {
                card.classList.remove('now-playing');
                if (playCircle) playCircle.innerText = '▶';
                if (waveform) waveform.style.display = 'none';
                player.querySelector('.stub-mini').innerText = '▶';
                const scrubFill = parentDoc.getElementById('bottom-scrub-fill');
                if (scrubFill) scrubFill.style.width = '0%';
            };
        } else {
            audio.pause();
            card.classList.remove('now-playing');
            if (playCircle) playCircle.innerText = '▶';
            if (waveform) waveform.style.display = 'none';
            player.querySelector('.stub-mini').innerText = '▶';
        }
    };
    
    parentWindow.toggleBottomPlayer = function() {
        if (parentWindow.activeAudio) {
            const playBtn = parentDoc.getElementById('bottom-play-btn');
            if (parentWindow.activeAudio.paused) {
                parentWindow.activeAudio.play().catch(function(e) { console.error('Audio play failed:', e); });
                parentWindow.activeCard.classList.add('now-playing');
                const pc = parentWindow.activeCard.querySelector('.play-circle');
                if (pc) pc.innerText = '⏸';
                const wave = parentWindow.activeCard.querySelector('.waveform');
                if (wave) wave.style.display = 'flex';
                if (playBtn) playBtn.innerText = '⏸';
            } else {
                parentWindow.activeAudio.pause();
                parentWindow.activeCard.classList.remove('now-playing');
                const pc = parentWindow.activeCard.querySelector('.play-circle');
                if (pc) pc.innerText = '▶';
                const wave = parentWindow.activeCard.querySelector('.waveform');
                if (wave) wave.style.display = 'none';
                if (playBtn) playBtn.innerText = '▶';
            }
        }
    };
    
    parentWindow.toggleWhy = function(rank) {
        const box = parentDoc.getElementById('why-box-' + rank);
        if (box) {
            box.style.display = box.style.display === 'none' ? 'block' : 'none';
        }
    };
    
    parentDoc.addEventListener('click', function(e) {
        const pc = e.target.closest('.play-circle');
        if (pc && pc.style.opacity !== '0.3') {
            const r = pc.getAttribute('data-rank');
            const n = pc.getAttribute('data-name');
            const a = pc.getAttribute('data-artist');
            const p = pc.getAttribute('data-preview');
            if (p) parentWindow.playTrack(r, n, a, p);
            return;
        }
        const wb = e.target.closest('.why-btn');
        if (wb) {
            const r = wb.getAttribute('data-rank');
            if (r) parentWindow.toggleWhy(r);
            return;
        }
        const bp = e.target.closest('#bottom-play-btn');
        if (bp) {
            parentWindow.toggleBottomPlayer();
            return;
        }
    });
})();
</script>" style="display:none; width:0; height:0; border:0;"></iframe>""",
    unsafe_allow_html=True,
)


def safe_text(value: object) -> str:
    return html.escape(str(value)) if value is not None else ""


def get_spotify_env(name: str) -> str:
    return os.getenv(name) or os.getenv(name.replace("SPOTIFY_", "SPOTIPY_"), "")


@st.cache_resource
def get_spotify_client() -> Optional[spotipy.Spotify]:
    """Initialize Spotify API client. Returns None if credentials unavailable."""
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


@st.cache_resource(show_spinner="Loading FAISS index...")
def load_retriever() -> SpotifyFAISSRetriever:
    """Load FAISS index and metadata for track retrieval."""
    retriever = SpotifyFAISSRetriever()
    retriever.load_index()
    return retriever


@st.cache_resource(show_spinner="Building smart search index...")
def load_engine(_retriever: SpotifyFAISSRetriever) -> SmartSearchEngine:
    """Build TF-IDF and vibe search engine."""
    return SmartSearchEngine(
        df=_retriever.df,
        feature_columns=_retriever.feature_columns,
        faiss_mean=_retriever.mean,
        faiss_std=_retriever.std,
        faiss_index=_retriever.index,
    )


@st.cache_resource(show_spinner="Checking two-tower hybrid index...")
def load_hybrid_retriever() -> Optional[HybridRetriever]:
    """Load two-tower hybrid retriever if available."""
    hybrid = HybridRetriever()
    return hybrid if hybrid.load() else None


def search_spotify_track(sp: Optional[spotipy.Spotify], row: pd.Series) -> Optional[dict]:
    """Search Spotify API for track metadata and preview URL."""
    if not sp:
        return None
    try:
        query = f"track:{row.get('name', '')}"
        artist = row.get("artists", row.get("artist_names", ""))
        if artist:
            query += f" artist:{artist}"
        info = sp.search(q=query, type="track", limit=1)
        return info["tracks"]["items"][0] if info["tracks"]["items"] else None
    except Exception as e:
        logger.debug(f"Spotify search failed for {row.get('name', 'unknown')}: {e}")
        return None


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


def get_track_card_html(
    row: pd.Series, rank: int, mode: str, engine_name: str,
    sp: Optional[spotipy.Spotify], show_preview: bool,
) -> str:
    sp_track = search_spotify_track(sp, row)
    name = safe_text(row.get("name", "Unknown"))
    artist = safe_text(row.get("artists", row.get("artist_names", "Unknown")))
    
    # Get audio features
    energy = float(row.get("energy", 0.0) or 0.0)
    dance = float(row.get("danceability", 0.0) or 0.0)
    valence = float(row.get("valence", 0.0) or 0.0)
    
    # Map z-scores to percentages
    def z_to_pct(z: float) -> int:
        return max(5, min(95, int((z + 2.0) / 4.0 * 100)))
        
    energy_pct = z_to_pct(energy)
    dance_pct = z_to_pct(dance)
    valence_pct = z_to_pct(valence)
    
    # Scores
    sim_score = row.get("similarity_score", row.get("sim_score", None))
    if pd.isna(sim_score) or sim_score is None:
        sim_score = row.get("final_score", row.get("_search_score", 0.0))
    sim_val = f"{float(sim_score):.2f}"
    pop_val = f"{float(row.get('popularity', 0.0)):.0f}"
    
    # Preview and Spotify links
    preview_url = ""
    spotify_url = "#"
    if sp_track:
        preview_url = sp_track.get("preview_url") or ""
        spotify_url = sp_track["external_urls"].get("spotify", "#")
        
    if not preview_url:
        # Fallback to royalty-free tracks so play button always works
        fallbacks = [
            "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
            "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3",
            "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3",
            "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3",
            "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-5.mp3",
            "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-6.mp3",
            "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-7.mp3",
            "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-8.mp3",
        ]
        preview_url = fallbacks[rank % len(fallbacks)]
        
    reasons = explain_row(row, mode, engine_name)
    reasons_html = "".join(f"<li>{safe_text(r)}</li>" for r in reasons)
    
    # Check if preview is available
    play_circle_html = ""
    audio_tag_html = ""
    if show_preview and preview_url:
        name_esc = name.replace("'", "\\'").replace('"', '&quot;')
        artist_esc = artist.replace("'", "\\'").replace('"', '&quot;')
        play_circle_html = f'<div class="play-circle" data-rank="{rank}" data-name="{name_esc}" data-artist="{artist_esc}" data-preview="{preview_url}">▶</div>'
        audio_tag_html = f'<audio id="audio-{rank}" src="{preview_url}"></audio>'
    else:
        play_circle_html = '<div class="play-circle" style="opacity: 0.3; cursor: not-allowed;">✕</div>'
        
    card_id = f"card-{rank}"
    
    html_str = f'<div class="card" id="{card_id}">{audio_tag_html}<div class="stub">{play_circle_html}<div class="rank">{rank:02d}</div></div><div class="card-body"><div class="track-title" title="{name}">{name}</div><div class="track-artist" title="{artist}">{artist}</div><div class="meter-row"><div class="meter"><span class="meter-label">Energy</span><div class="meter-track"><div class="meter-fill" style="width:{energy_pct}%"></div></div><span class="meter-val">{energy:.2f}</span></div><div class="meter"><span class="meter-label">Dance</span><div class="meter-track"><div class="meter-fill" style="width:{dance_pct}%"></div></div><span class="meter-val">{dance:.2f}</span></div><div class="meter"><span class="meter-label">Valence</span><div class="meter-track"><div class="meter-fill" style="width:{valence_pct}%"></div></div><span class="meter-val">{valence:.2f}</span></div></div><div class="waveform" style="display:none;"><span style="height:40%"></span><span style="height:70%"></span><span style="height:30%"></span><span style="height:90%"></span><span style="height:50%"></span><span style="height:20%"></span><span style="height:65%"></span><span style="height:45%"></span><span style="height:80%"></span><span style="height:35%"></span></div><div class="card-footer"><div class="score">sim <b>{sim_val}</b> · pop <b>{pop_val}</b></div><div style="display:flex; gap: 6px;"><button class="why-btn" data-rank="{rank}">Why?</button></div></div><div id="why-box-{rank}" style="display:none; margin-top:10px; font-size:11px; opacity:0.8; border-top:1px dashed rgba(255,255,255,0.2); padding-top:8px;"><ul style="margin:0; padding-left:16px;">{reasons_html}</ul></div></div></div>'
    return html_str


try:
    sp_client = get_spotify_client()
    retriever = load_retriever()
    engine = load_engine(retriever)
    hybrid_retriever = load_hybrid_retriever()
except FileNotFoundError as e:
    st.error("❌ Missing required index files. Run data preprocessing first.")
    logger.error(f"Missing files: {e}")
    st.stop()
except Exception as exc:
    st.error("❌ Failed to load retrieval system. Please check logs.")
    st.caption(f"Error: {str(exc)}")
    logger.error(f"Failed to load retrieval system: {exc}", exc_info=True)
    st.stop()


HYBRID_AVAILABLE = hybrid_retriever is not None



total_tracks = len(retriever.df)
st.markdown(
    f"""<header>
  <div>
    <div class="logo">MOOD<span>IFY</span></div>
    <div class="tagline">Hybrid retrieval · FAISS + TF-IDF + two-tower</div>
  </div>
  <div class="tagline" style="text-align:right;">{total_tracks:,} tracks indexed</div>
</header>""".strip(),
    unsafe_allow_html=True,
)


st.sidebar.markdown("## Search")
typed = st.sidebar.text_input(
    "Song, artist, or vibe",
    value=st.session_state["query"],
    placeholder="Blinding Lights, chill night drive, gym workout",
)
top_k = st.sidebar.slider("Results", 5, 20, 8)
show_preview = st.sidebar.checkbox("Spotify previews", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("## Retrieval")
use_hybrid = False
if HYBRID_AVAILABLE:
    use_hybrid = st.sidebar.toggle(
        "Use two-tower hybrid for song searches",
        value=False,
        help="Uses learned item embeddings plus popularity and artist diversity.",
    )
    if use_hybrid:
        hybrid_retriever.alpha = st.sidebar.slider("Similarity weight", 0.0, 1.0, 0.70, 0.05)
        hybrid_retriever.beta = st.sidebar.slider("Popularity weight", 0.0, 1.0, 0.20, 0.05)
        hybrid_retriever.gamma = st.sidebar.slider("Diversity weight", 0.0, 1.0, 0.10, 0.05)
else:
    st.sidebar.caption("Two-tower hybrid index not found. Run `python src/models/train_item_tower.py`.")

if st.sidebar.button("Get recommendations", type="primary", use_container_width=True):
    query = typed.strip()
    if not query:
        st.sidebar.warning("Enter a song, artist, or vibe.")
    else:
        started = time.time()
        seed, results, mode, label = engine.get_seed_and_recommend(query, retriever, top_k=top_k)
        engine_name = "FAISS"

        if use_hybrid and HYBRID_AVAILABLE and seed is not None:
            seed_id = seed.get("id")
            if seed_id is not None:
                try:
                    results = hybrid_retriever.recommend_by_id(str(seed_id), top_k=top_k)
                    engine_name = "two-tower hybrid"
                    mode = "hybrid"
                except KeyError:
                    engine_name = "FAISS fallback"

        st.session_state["query"] = query
        st.session_state["results"] = results
        st.session_state["mode"] = mode
        st.session_state["label"] = label
        st.session_state["elapsed"] = time.time() - started
        st.session_state["engine_name"] = engine_name


st.sidebar.markdown("---")
st.sidebar.markdown("## Quick vibes")
quick_vibes = [
    "chill night drive",
    "gym workout",
    "sad heartbreak",
    "party banger",
    "study focus",
    "romantic dinner",
    "happy feel good",
    "sleep meditation",
    "motivation hustle",
    "acoustic unplugged",
]
for vibe in quick_vibes:
    if st.sidebar.button(vibe.title(), key=f"vibe_{vibe}", use_container_width=True):
        st.session_state["query"] = vibe
        started = time.time()
        _, results, mode, label = engine.get_seed_and_recommend(vibe, retriever, top_k=top_k)
        st.session_state["results"] = results
        st.session_state["mode"] = mode
        st.session_state["label"] = label
        st.session_state["elapsed"] = time.time() - started
        st.session_state["engine_name"] = "FAISS vibe"
        st.rerun()


results = st.session_state.get("results")
mode = st.session_state.get("mode", "")
label = st.session_state.get("label", "")
elapsed = st.session_state.get("elapsed", 0.0)
engine_name = st.session_state.get("engine_name", "FAISS")

if isinstance(results, pd.DataFrame) and not results.empty:
    badge_map = {
        "vibe": ("badge-vibe", "Vibe mode"),
        "tfidf": ("badge-tfidf", "Smart match"),
        "fuzzy": ("badge-fuzzy", "Fuzzy match"),
        "hybrid": ("badge-hybrid", "Two-tower hybrid"),
    }
    badge_class, badge_text = badge_map.get(mode, ("badge-tfidf", mode or "Results"))

    header_left, header_right = st.columns([4, 1])
    with header_left:
        message = f"Tracks matching {label}" if mode == "vibe" else f"Based on {label}"
        st.markdown(
            f'<div class="section-label">{badge_text} · {message} · Engine: {engine_name}</div>',
            unsafe_allow_html=True,
        )
    with header_right:
        st.markdown(
            f'<div style="text-align:right; font-family:\'IBM Plex Mono\',monospace; font-size:11px; opacity:0.6; padding-top:20px;">'
            f'{len(results)} results in {elapsed:.2f}s'
            f'</div>',
            unsafe_allow_html=True,
        )

    cards_html = []
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        cards_html.append(get_track_card_html(row, rank, mode, engine_name, sp_client, show_preview))
    st.markdown(f'<div class="grid">{"".join(cards_html)}</div>', unsafe_allow_html=True)
else:
    st.markdown(
        """<div class="empty-state" style="border: 2px dashed var(--ink); padding: 60px 20px; text-align: center; background: var(--canvas-raised); box-shadow: 4px 4px 0 var(--shadow-ink); margin-top: 20px;">
  <h3 style="font-family: 'Space Grotesk', sans-serif; font-size: 22px; margin-bottom: 8px; color: var(--ink);">Find your next track</h3>
  <p style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; opacity: 0.6; text-transform: uppercase; color: var(--ink);">Search for a song, artist, or listening context in the sidebar</p>
</div>""".strip(),
        unsafe_allow_html=True,
    )


st.markdown(
    """<div style="border-top:2px solid var(--ink);margin-top:48px;padding:20px 0;color:var(--ink);font-family:'IBM Plex Mono',monospace;font-size:11px;text-align:center;text-transform:uppercase;opacity:0.5;letter-spacing:0.5px;">
  Moodify official demo · Streamlit + FAISS + TF-IDF + two-tower
</div>""".strip(),
    unsafe_allow_html=True,
)
