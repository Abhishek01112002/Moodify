# main.py
"""
Main entry point for Spotify data collection pipeline.

PHASE 1: Data Collection
This script collects Spotify track data from your playlists.
"""
from pipeline import run_pipeline
import os

if __name__ == "__main__":
    print("=" * 70)
    print("SPOTIFY DATA COLLECTION - PHASE 1")
    print("=" * 70)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("\n❌ Error: .env file not found!")
        print("   Please create a .env file with your Spotify credentials.")
        print("   See QUICKSTART.md for instructions.")
        exit(1)
    
    # Define your playlists here
    # Replace with your actual Spotify playlist IDs
    # To get playlist ID: Share playlist → Copy link → Use the ID at the end
    playlists = {
        "sad": "7dd7kIZf6QA7gMtfgjqWGv",
        "romantic": "55PVuXcePN1SJUh8yczGuR",
        "party": "4F9XjRMCeyxlmysK16V85W",
        "chill": "04qR7ZNAXWilCfoZUNg3VH",
        "old_melody": "4A3nkKd93i7OcIjmxWBmZw"
    }
    
    # Check if playlists are configured
    if any("YOUR_" in pid for pid in playlists.values()):
        print("\n⚠️  Warning: Please update playlist IDs in main.py")
        print("\nTo get playlist IDs:")
        print("  1. Open Spotify (web or desktop)")
        print("  2. Go to your playlist")
        print("  3. Click 'Share' → 'Copy link to playlist'")
        print("  4. The playlist ID is the last part of the URL")
        print("     Example: https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M")
        print("     Playlist ID: 37i9dQZF1DXcBWIGoYBM5M")
        print("\nAfter updating, run this script again.")
        exit(0)
    
    # Run the pipeline
    print("\nStarting data collection...")
    print("Note: First run will open browser for Spotify authorization")
    
    try:
        df = run_pipeline(playlists, output_file="labeled_spotify_songs.csv")
        
        if df is not None and len(df) > 0:
            print("\n" + "=" * 70)
            print("✅ DATA COLLECTION COMPLETE!")
            print("=" * 70)
            print(f"\n📊 Collected {len(df)} tracks")
            print(f"💾 Saved to: labeled_spotify_songs.csv")
            print(f"\n📈 Next step: Run preprocessing and EDA")
            print("   python run_phase2_phase3.py")
        else:
            print("\n❌ No data collected. Please check your playlist IDs and try again.")
    except Exception as e:
        print(f"\n❌ Error during data collection: {e}")
        print("\nTroubleshooting:")
        print("  - Check your .env file has correct credentials")
        print("  - Verify playlist IDs are correct")
        print("  - Make sure you have internet connection")
        print("  - See QUICKSTART.md for detailed help")
