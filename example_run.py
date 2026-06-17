"""
Example script showing complete workflow from data collection to EDA.
This is a reference implementation - modify as needed.
"""
import os
import sys

def main():
    print("=" * 70)
    print("SPOTIFY RECOMMENDATION SYSTEM - COMPLETE WORKFLOW EXAMPLE")
    print("=" * 70)
    
    # Check prerequisites
    print("\n[Step 0] Checking prerequisites...")
    
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("   Create .env with your Spotify API credentials")
        print("   See QUICKSTART.md for instructions")
        return
    
    print("✅ .env file found")
    
    # PHASE 1: Data Collection
    print("\n" + "=" * 70)
    print("PHASE 1: DATA COLLECTION")
    print("=" * 70)
    
    from pipeline import run_pipeline
    
    # Example playlists - REPLACE WITH YOUR PLAYLIST IDs
    playlists = {
        "sad": "YOUR_SAD_PLAYLIST_ID",
        "romantic": "YOUR_ROMANTIC_PLAYLIST_ID",
        "party": "YOUR_PARTY_PLAYLIST_ID",
        "chill": "YOUR_CHILL_PLAYLIST_ID",
        "old_melody": "YOUR_OLD_MELODY_PLAYLIST_ID"
    }
    
    # Check if configured
    if any("YOUR_" in pid for pid in playlists.values()):
        print("\n⚠️  Please update playlist IDs in this script first!")
        print("   Edit example_run.py and replace YOUR_*_PLAYLIST_ID with actual IDs")
        return
    
    print("Collecting data from Spotify...")
    df = run_pipeline(playlists, output_file="labeled_spotify_songs.csv")
    
    if df is None or len(df) == 0:
        print("❌ Data collection failed. Exiting.")
        return
    
    print(f"✅ Collected {len(df)} tracks")
    
    # PHASE 2: Data Preprocessing
    print("\n" + "=" * 70)
    print("PHASE 2: DATA PREPROCESSING")
    print("=" * 70)
    
    from data_preprocessing import preprocess_dataset
    
    print("Cleaning and preprocessing data...")
    df_clean, scaler = preprocess_dataset(
        df, 
        normalize_method='standard', 
        encode_genres=True
    )
    
    df_clean.to_csv("spotify_songs_cleaned.csv", index=False)
    print(f"✅ Saved cleaned data: spotify_songs_cleaned.csv")
    
    # PHASE 3: EDA
    print("\n" + "=" * 70)
    print("PHASE 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    
    from eda import generate_eda_report
    
    print("Generating EDA visualizations...")
    generate_eda_report(df_clean, output_dir="eda_output", mood_col='mood_label')
    
    print("\n" + "=" * 70)
    print("✅ ALL PHASES COMPLETE!")
    print("=" * 70)
    print("\n📁 Output files:")
    print("  - labeled_spotify_songs.csv (raw data)")
    print("  - spotify_songs_cleaned.csv (preprocessed data)")
    print("  - eda_output/ (visualizations)")
    print("\n🎯 Next: Proceed to PHASE 4 (Build recommendation engines)")

if __name__ == "__main__":
    main()

