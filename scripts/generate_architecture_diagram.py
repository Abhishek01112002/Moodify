"""
Generate high-fidelity architecture diagram for Moodify.
Outputs both PNG and SVG to screenshots/ for bulletproof GitHub README display.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=220)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 90)
    ax.axis('off')

    # Color Palette - Spotify Modern Theme
    bg_card = '#161b22'
    border_card = '#30363d'
    border_accent = '#1db954'
    text_main = '#f0f6fc'
    text_muted = '#8b949e'
    text_accent = '#1db954'
    text_sub = '#58a6ff'

    # Title Header
    ax.text(80, 84.8, "MOODIFY SYSTEM ARCHITECTURE", 
            fontsize=21, fontweight='bold', color=text_main, ha='center', va='center',
            family='sans-serif')
    ax.text(80, 81.2, "End-to-End Hybrid Music Recommendation & Retrieval Engine", 
            fontsize=12, color=text_muted, ha='center', va='center',
            family='sans-serif')

    def draw_card(x, y, w, h, title, subtitle="", items=None, accent=False):
        edge = border_accent if accent else border_card
        lw = 2.0 if accent else 1.2
        
        # Outer Card Container
        rect = patches.FancyBboxPatch((x, y), w, h,
                                      boxstyle="round,pad=0.5,rounding_size=1.2",
                                      facecolor=bg_card, edgecolor=edge, linewidth=lw)
        ax.add_patch(rect)
        
        # Header text
        ax.text(x + 2.2, y + h - 3.2, title, 
                fontsize=11.5, fontweight='bold', color=text_accent if accent else text_main,
                ha='left', va='center', family='sans-serif')
        
        # Subtitle / Category
        curr_y = y + h - 6.2
        if subtitle:
            ax.text(x + 2.2, curr_y, subtitle, 
                    fontsize=8.8, fontstyle='italic', color=text_sub if not accent else '#34d399',
                    ha='left', va='center', family='sans-serif')
            curr_y -= 3.6

        # Feature Bullet Points
        if items:
            for it in items:
                ax.text(x + 2.5, curr_y, f"• {it}", 
                        fontsize=8.6, color=text_main, ha='left', va='center',
                        family='sans-serif')
                curr_y -= 2.7

    def draw_arrow(x1, y1, x2, y2, color='#58a6ff'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->,head_width=0.42,head_length=0.65",
                                    color=color, lw=1.8, shrinkA=3, shrinkB=3,
                                    connectionstyle="arc3,rad=0"))

    # Stage Headers
    ax.text(20, 75.5, "1. DATA INGESTION", fontsize=10.5, fontweight='bold', color='#58a6ff', ha='center', family='sans-serif')
    ax.text(52, 75.5, "2. PREPROCESSING & EDA", fontsize=10.5, fontweight='bold', color='#58a6ff', ha='center', family='sans-serif')
    ax.text(85, 75.5, "3. MULTI-STAGE RETRIEVAL", fontsize=10.5, fontweight='bold', color='#58a6ff', ha='center', family='sans-serif')
    ax.text(117, 75.5, "4. HYBRID RERANKING", fontsize=10.5, fontweight='bold', color='#58a6ff', ha='center', family='sans-serif')
    ax.text(147, 75.5, "5. USER INTERFACE", fontsize=10.5, fontweight='bold', color='#58a6ff', ha='center', family='sans-serif')

    # Column 1: Data Ingestion (x: 8 to 32, width 24)
    draw_card(8, 52, 24, 20, "Spotify API", 
              subtitle="Spotipy Client Ingestion",
              items=["12+ Audio Features", "Real-Time Track Metadata", "Playlist Harvesting", "Artist Popularity & Genres"])
    
    draw_card(8, 28, 24, 20, "Kaggle Catalog", 
              subtitle="Scalable Benchmark Corpus",
              items=["High-Volume Track Set", "Genre & Mood Ground-Truth", "Million-Song Features", "Offline Evaluation Grounding"])

    # Column 2: Preprocessing & EDA (x: 40 to 64, width 24)
    draw_card(40, 28, 24, 44, "Data Preprocessing", 
              subtitle="Feature Engineering Pipeline",
              items=[
                  "MinMax Feature Scaling",
                  "Outlier & Missing Handling",
                  "9D Audio Feature Vector",
                  "Correlation Heatmap (EDA)",
                  "Feature Variance Analysis",
                  "Fast Parquet Disk Caching",
                  "Proxy Mood & Genre Labels"
              ])

    # Column 3: Multi-Stage Retrieval Layer (x: 73 to 97, width 24)
    draw_card(73, 54, 24, 18, "Smart Search", 
              subtitle="NLP Text & Vibe Search",
              items=["TF-IDF Character N-grams", "RapidFuzz Typo-Tolerant", "Natural Language Vibe Map", "Sub-10ms Candidate Filtering"])

    draw_card(73, 32, 24, 18, "FAISS Vector Index", 
              subtitle="Fast Nearest Neighbors",
              items=["9D Normalized Audio Space", "Cosine & L2 Proximity", "Sub-3ms Query Latency", "91.8% Catalog Coverage"],
              accent=True)

    draw_card(73, 10, 24, 18, "Self-Supervised DL", 
              subtitle="SimCLR Item & Two-Tower",
              items=["Stochastic Feature Perturbation", "Contrastive NT-Xent Loss", "Dense Audio Embeddings", "Profile-Aware User Tower"],
              accent=True)

    # Column 4: Reranking & Hybrid Scoring (x: 106 to 128, width 22)
    draw_card(106, 22, 22, 50, "Hybrid Reranker", 
              subtitle="Multi-Objective Scoring",
              items=[
                  "Objective Function:",
                  "α × Similarity Score",
                  "β × Popularity Weight",
                  "γ × Artist Diversity",
                  "Prevents Filter Bubbles",
                  "De-duplicates Artists",
                  "Balances Serendipity",
                  "Explainable Breakdown",
                  "Interactive Tuning"
              ],
              accent=True)

    # Column 5: Serving & User Experience (x: 136 to 158, width 22)
    draw_card(136, 48, 22, 24, "Streamlit Demo", 
              subtitle="Official Live Web App",
              items=["Interactive Sliders", "Real-Time Recommendations", "Audio Preview Players", "Streamlit Cloud Deployed", "Live Public URL"],
              accent=True)

    draw_card(136, 18, 22, 24, "FastAPI Server", 
              subtitle="Modern SPA Web Engine",
              items=["RESTful API Endpoints", "Mood-Reactive Gradients", "LocalStorage Recent Search", "Audio Preview Engine", "Full-Stack Separation"])

    # Connecting Arrows
    # Data to Preprocessing
    draw_arrow(32, 62, 40, 56, color='#8b949e')
    draw_arrow(32, 38, 40, 44, color='#8b949e')

    # Preprocessing to Retrieval
    draw_arrow(64, 60, 73, 63, color='#58a6ff')
    draw_arrow(64, 50, 73, 41, color='#1db954')
    draw_arrow(64, 38, 73, 19, color='#1db954')

    # Retrieval to Reranker
    draw_arrow(97, 63, 106, 56, color='#58a6ff')
    draw_arrow(97, 41, 106, 47, color='#1db954')
    draw_arrow(97, 19, 106, 36, color='#1db954')

    # Reranker to Applications
    draw_arrow(128, 55, 136, 60, color='#1db954')
    draw_arrow(128, 38, 136, 30, color='#58a6ff')

    # Footer metrics bar
    footer_rect = patches.FancyBboxPatch((8, 3.2), 150, 4.4,
                                         boxstyle="round,pad=0.2,rounding_size=0.6",
                                         facecolor='#161b22', edgecolor='#30363d', linewidth=1.0)
    ax.add_patch(footer_rect)
    ax.text(83, 5.4, 
            "Offline Evaluation Highlights:  NDCG@10: 0.491  |  Precision@10: 0.472  |  Catalog Coverage: 91.8%  |  Query Latency: 2.08 ms  |  19 Unit Tests (CI)",
            fontsize=9.5, fontweight='bold', color='#1db954', ha='center', va='center', family='sans-serif')

    out_dir = Path("screenshots")
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / "architecture.png"
    svg_path = out_dir / "architecture.svg"

    plt.tight_layout()
    plt.savefig(png_path, dpi=240, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.savefig(svg_path, format='svg', facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Architecture diagrams generated:\n- {png_path}\n- {svg_path}")

if __name__ == "__main__":
    create_architecture_diagram()
