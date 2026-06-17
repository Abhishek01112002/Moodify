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
    page_title="Moodify",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background-color: #0b0d0f;
    color: #f3f4f6;
}
section[data-testid="stSidebar"] {
    background: #111418;
    border-right: 1px solid #23272f;
}
.hero {
    padding: 10px 0 18px 0;
    border-bottom: 1px solid #23272f;
    margin-bottom: 18px;
}
.hero h1 {
    margin: 0;
    color: #ffffff;
    font-size: 2.15rem;
    line-height: 1.1;
}
.hero p {
    margin: 8px 0 0 0;
    color: #aeb6c2;
    font-size: 0.95rem;
}
.metric-strip {
    display: flex;
    gap: 12px;
    justify-content: flex-end;
}
.mini-metric {
    min-width: 105px;
    border: 1px solid #252b33;
    background: #151a20;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
.mini-metric b {
    display: block;
    color: #1db954;
    font-size: 1.25rem;
}
.mini-metric span {
    color: #8b949e;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.badge {
    display: inline-flex;
    align-items: center;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-vibe { background: #6d28d9; color: #fff; }
.badge-tfidf { background: #1db954; color: #07110b; }
.badge-fuzzy { background: #d97706; color: #111; }
.badge-hybrid { background: #2563eb; color: #fff; }
.track-card {
    background: #151a20;
    border: 1px solid #252b33;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.track-card:hover {
    border-color: #334155;
    background: #171d24;
}
.track-name {
    margin: 0 0 2px 0;
    color: #ffffff;
    font-size: 1.02rem;
    font-weight: 700;
}
.track-artist {
    margin: 0 0 8px 0;
    color: #aeb6c2;
    font-size: 0.86rem;
}
.pop-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 6px;
}
.pop-bar-bg {
    flex: 1;
    background: #29313b;
    border-radius: 99px;
    height: 4px;
}
.pop-bar-fg {
    background: #1db954;
    border-radius: 99px;
    height: 4px;
}
.muted {
    color: #8b949e;
    font-size: 0.78rem;
}
.chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 8px;
}
.chip {
    border: 1px solid #2d3744;
    background: #10151b;
    color: #cbd5e1;
    border-radius: 999px;
    padding: 3px 9px;
    font-size: 0.72rem;
}
.why {
    margin-top: 10px;
    color: #cbd5e1;
    font-size: 0.82rem;
    line-height: 1.55;
}
.why strong {
    color: #ffffff;
}
.empty-state {
    text-align: center;
    padding: 70px 20px;
    color: #aeb6c2;
}
</style>
""",
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


def render_track_card(
    row: pd.Series, rank: int, mode: str, engine_name: str,
    sp: Optional[spotipy.Spotify], show_preview: bool,
) -> None:
    sp_track = search_spotify_track(sp, row)
    name = safe_text(row.get("name", "Unknown"))
    artist = safe_text(row.get("artists", row.get("artist_names", "Unknown")))
    pop = float(row.get("popularity", 0) or 0)
    score = float(row.get("final_score", row.get("_search_score", 0)) or 0)

    st.markdown('<div class="track-card">', unsafe_allow_html=True)
    image_col, info_col, action_col = st.columns([0.85, 4.2, 1.8])

    with image_col:
        if sp_track and sp_track["album"]["images"]:
            st.image(sp_track["album"]["images"][0]["url"], width=104)
        else:
            st.markdown(
                '<div style="width:104px;height:104px;border-radius:8px;'
                'background:#29313b;display:flex;align-items:center;'
                'justify-content:center;color:#8b949e;font-weight:700;">ART</div>',
                unsafe_allow_html=True,
            )

    with info_col:
        st.markdown(
            f'<p class="track-name">{rank}. {name}</p>'
            f'<p class="track-artist">{artist}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="pop-wrap">'
            f'<div class="pop-bar-bg"><div class="pop-bar-fg" style="width:{max(2, min(100, pop))}%"></div></div>'
            f'<span class="muted">Pop {pop:.0f}</span>'
            f'<span class="muted">Score {score:.3f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        chips = []
        for label, column in [("Energy", "energy"), ("Mood", "valence"), ("Dance", "danceability")]:
            if pd.notna(row.get(column, None)):
                chips.append(f'<span class="chip">{label}: {float(row[column]):.2f}</span>')
        if chips:
            st.markdown(f'<div class="chips">{"".join(chips)}</div>', unsafe_allow_html=True)

        reasons = explain_row(row, mode, engine_name)
        if reasons:
            reason_html = "<br>".join(f"- {safe_text(reason)}" for reason in reasons)
            st.markdown(
                f'<div class="why"><strong>Why this song?</strong><br>{reason_html}</div>',
                unsafe_allow_html=True,
            )

    with action_col:
        if sp_track:
            if show_preview and sp_track.get("preview_url"):
                st.audio(sp_track["preview_url"], format="audio/mp3")
            st.link_button(
                "Open Spotify",
                sp_track["external_urls"].get("spotify", "#"),
                use_container_width=True,
            )
        else:
            st.button(
                "Preview unavailable",
                disabled=True,
                key=f"disabled_{rank}_{row.get('id', row.get('track_id', rank))}",
                use_container_width=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


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

for key, value in {
    "query": "",
    "results": None,
    "mode": "",
    "label": "",
    "elapsed": 0.0,
    "engine_name": "FAISS",
}.items():
    st.session_state.setdefault(key, value)


left, right = st.columns([3.2, 1.2])
with left:
    st.markdown(
        """
        <div class="hero">
          <h1>Moodify</h1>
          <p>Hybrid Spotify-style recommendations with smart search, vibe search, FAISS retrieval, and explainable ranking.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    total_tracks = len(retriever.df)
    engine_count = 4 if HYBRID_AVAILABLE else 3
    st.markdown(
        f"""
        <div class="metric-strip">
          <div class="mini-metric"><b>{total_tracks:,}</b><span>Tracks</span></div>
          <div class="mini-metric"><b>{engine_count}</b><span>Engines</span></div>
        </div>
        """,
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
        st.markdown(f'<span class="badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)
        st.caption(f"{message} | Engine: {engine_name}")
    with header_right:
        st.caption(f"{len(results)} results in {elapsed:.2f}s")

    st.markdown("")
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        render_track_card(row, rank, mode, engine_name, sp_client, show_preview)
else:
    st.markdown(
        """
        <div class="empty-state">
          <h3>Find your next track</h3>
          <p>Search for a song, artist, or listening context like "chill night drive" or "gym workout".</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div style="border-top:1px solid #23272f;margin-top:28px;padding:18px 0;color:#8b949e;font-size:0.76rem;text-align:center;">
      Moodify official demo | Streamlit + FAISS + TF-IDF + fuzzy search + vibe retrieval
    </div>
    """,
    unsafe_allow_html=True,
)
