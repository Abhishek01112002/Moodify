"""
Simple Music Recommendation System
Uses similarity-based recommendations based on mood, genre, and audio features
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys

def load_data():
    """Load the cleaned dataset"""
    df = pd.read_csv('spotify_songs_cleaned.csv')
    return df

def get_track_features(df, track_id):
    """Get feature vector for a track"""
    track = df[df['track_id'] == track_id]
    if track.empty:
        return None
    
    # Select numerical features for similarity (genres, audio features, popularity)
    feature_cols = [col for col in df.columns if 'genre_' in col]
    feature_cols.extend(['artist_popularity', 'popularity'])
    
    return track[feature_cols].values[0]

def recommend_by_mood(df, mood_label, n_recommendations=5):
    """Get recommendations for a specific mood"""
    mood_tracks = df[df['mood_label'] == mood_label]
    
    if mood_tracks.empty:
        print(f"❌ No tracks found for mood: {mood_label}")
        return None
    
    print(f"\n🎵 Top {n_recommendations} Recommendations for '{mood_label}' mood:")
    print("=" * 80)
    
    # Show top tracks by popularity
    top_tracks = mood_tracks.nlargest(n_recommendations, 'popularity')[
        ['name', 'artist_names', 'primary_genre', 'popularity', 'mood_label']
    ]
    
    for idx, (_, row) in enumerate(top_tracks.iterrows(), 1):
        print(f"\n{idx}. {row['name']}")
        print(f"   Artist: {row['artist_names']}")
        print(f"   Genre: {row['primary_genre']}")
        print(f"   Popularity: {row['popularity']:.2f}")
    
    return top_tracks

def recommend_similar_tracks(df, track_name, n_recommendations=5):
    """Get recommendations similar to a given track"""
    track = df[df['name'].str.lower() == track_name.lower()]
    
    if track.empty:
        print(f"❌ Track '{track_name}' not found in database")
        print("\n💡 Available tracks:")
        print(df['name'].head(10).tolist())
        return None
    
    track_id = track['track_id'].values[0]
    mood = track['mood_label'].values[0]
    
    print(f"\n🎵 Recommendations similar to: '{track_name}'")
    print(f"   Mood: {mood}")
    print("=" * 80)
    
    # Get feature vector for the input track
    feature_cols = [col for col in df.columns if 'genre_' in col]
    feature_cols.extend(['artist_popularity', 'popularity'])
    
    input_features = track[feature_cols].values
    all_features = df[feature_cols].values
    
    # Calculate similarity
    similarities = cosine_similarity(input_features, all_features)[0]
    
    # Get top similar tracks (excluding the input track itself)
    similar_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]
    
    recommendations = df.iloc[similar_indices][
        ['name', 'artist_names', 'primary_genre', 'popularity', 'mood_label', 'similarity']
    ].copy()
    recommendations['similarity'] = similarities[similar_indices]
    
    for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
        print(f"\n{idx}. {row['name']}")
        print(f"   Artist: {row['artist_names']}")
        print(f"   Genre: {row['primary_genre']}")
        print(f"   Mood: {row['mood_label']}")
        print(f"   Similarity: {row['similarity']:.2%}")
    
    return recommendations

def show_mood_stats(df):
    """Show statistics about each mood"""
    print("\n📊 Mood Statistics")
    print("=" * 80)
    
    for mood in df['mood_label'].unique():
        mood_df = df[df['mood_label'] == mood]
        print(f"\n{mood.upper()}:")
        print(f"  Total tracks: {len(mood_df)}")
        print(f"  Avg popularity: {mood_df['popularity'].mean():.2f}")
        print(f"  Avg artist followers: {mood_df['artist_followers'].mean():,.0f}")
        print(f"  Top genre: {mood_df['primary_genre'].mode()[0] if not mood_df['primary_genre'].mode().empty else 'Unknown'}")

def interactive_recommendations(df):
    """Interactive recommendation system"""
    while True:
        print("\n" + "=" * 80)
        print("🎵 SPOTIFY RECOMMENDATION SYSTEM")
        print("=" * 80)
        print("\n1. Get recommendations by mood")
        print("2. Find similar tracks")
        print("3. View mood statistics")
        print("4. Browse all tracks")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            print("\nAvailable moods:")
            moods = sorted(df['mood_label'].unique())
            for i, mood in enumerate(moods, 1):
                count = len(df[df['mood_label'] == mood])
                print(f"  {i}. {mood} ({count} tracks)")
            
            mood_input = input("\nEnter mood: ").strip()
            if mood_input in moods:
                recommend_by_mood(df, mood_input, n_recommendations=10)
            else:
                print(f"❌ Invalid mood. Choose from: {', '.join(moods)}")
        
        elif choice == '2':
            track_name = input("\nEnter track name (or partial): ").strip()
            matching = df[df['name'].str.contains(track_name, case=False, na=False)]
            
            if matching.empty:
                print(f"❌ No tracks found matching '{track_name}'")
            elif len(matching) > 1:
                print(f"\nFound {len(matching)} matches:")
                for i, (_, row) in enumerate(matching.iterrows(), 1):
                    print(f"  {i}. {row['name']} - {row['artist_names']}")
                idx = int(input("\nSelect track number: ")) - 1
                if 0 <= idx < len(matching):
                    track_name = matching.iloc[idx]['name']
                    recommend_similar_tracks(df, track_name, n_recommendations=10)
                else:
                    print("❌ Invalid selection")
            else:
                recommend_similar_tracks(df, track_name, n_recommendations=10)
        
        elif choice == '3':
            show_mood_stats(df)
        
        elif choice == '4':
            print("\nAll tracks in database:")
            print("=" * 80)
            display_df = df[['name', 'artist_names', 'mood_label', 'popularity']].copy()
            display_df.index = range(1, len(display_df) + 1)
            print(display_df.to_string())
        
        elif choice == '5':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-5")

if __name__ == "__main__":
    print("Loading dataset...")
    df = load_data()
    print(f"✅ Loaded {len(df)} songs from {df['mood_label'].nunique()} moods\n")
    
    # Check if arguments provided
    if len(sys.argv) > 1:
        if sys.argv[1] == 'mood' and len(sys.argv) > 2:
            mood = sys.argv[2]
            n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            recommend_by_mood(df, mood, n_recommendations=n)
        elif sys.argv[1] == 'similar' and len(sys.argv) > 2:
            track = ' '.join(sys.argv[2:])
            recommend_similar_tracks(df, track, n_recommendations=10)
        elif sys.argv[1] == 'stats':
            show_mood_stats(df)
    else:
        # Interactive mode
        interactive_recommendations(df)
