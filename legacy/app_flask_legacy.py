"""
Flask Web App for Spotify Recommendations with Album Art
[LEGACY] — Superseded by Streamlit demo at app/main.py

This Flask implementation is archived for reference only.
For the official demo, use: poetry run streamlit run app/main.py
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'spotify-recommendation-secret'

# Load Spotify credentials
CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')

# Initialize Spotify client
sp = None
try:
    if CLIENT_ID and CLIENT_SECRET:
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET
        ))
except:
    sp = None

# Load data
df = pd.read_csv('spotify_songs_cleaned.csv')
track_cache = {}

def get_track_image(track_id):
    """Get album art from Spotify API"""
    if track_id in track_cache:
        return track_cache[track_id]
    
    if not sp:
        return 'https://via.placeholder.com/300x300?text=Album+Art'
    
    try:
        track = sp.track(track_id)
        image_url = track['album']['images'][0]['url'] if track['album']['images'] else None
        if image_url:
            track_cache[track_id] = image_url
            return image_url
    except:
        pass
    
    return 'https://via.placeholder.com/300x300?text=Album+Art'

@app.route('/')
def index():
    moods = sorted(df['mood_label'].unique())
    mood_counts = {mood: len(df[df['mood_label'] == mood]) for mood in moods}
    return render_template('index.html', moods=moods, mood_counts=mood_counts)

@app.route('/api/recommendations/mood/<mood>')
def get_mood_recommendations(mood):
    """Get top recommendations for a mood"""
    mood_df = df[df['mood_label'] == mood].nlargest(10, 'popularity')
    
    recommendations = []
    for _, row in mood_df.iterrows():
        recommendations.append({
            'name': row['name'],
            'artist': row['artist_names'] if isinstance(row['artist_names'], str) else ', '.join(row['artist_names']),
            'genre': row['primary_genre'],
            'popularity': float(row['popularity']),
            'mood': row['mood_label'],
            'track_id': row['track_id'],
            'image': get_track_image(row['track_id']),
            'year': int(row['release_year']) if pd.notna(row['release_year']) else 'Unknown'
        })
    
    return jsonify(recommendations)

@app.route('/api/recommendations/similar/<track_name>')
def get_similar_recommendations(track_name):
    """Get recommendations similar to a track"""
    track = df[df['name'].str.lower() == track_name.lower()]
    
    if track.empty:
        # Try partial match
        track = df[df['name'].str.contains(track_name, case=False, na=False)].head(1)
    
    if track.empty:
        return jsonify({'error': 'Track not found'}), 404
    
    track_id = track['track_id'].values[0]
    mood = track['mood_label'].values[0]
    
    # Get feature vector
    feature_cols = [col for col in df.columns if 'genre_' in col]
    feature_cols.extend(['artist_popularity', 'popularity'])
    
    input_features = track[feature_cols].values
    all_features = df[feature_cols].values
    
    # Calculate similarity
    similarities = cosine_similarity(input_features, all_features)[0]
    
    # Get top similar tracks (excluding input)
    similar_indices = np.argsort(similarities)[::-1][1:11]
    
    recommendations = []
    for idx in similar_indices:
        row = df.iloc[idx]
        recommendations.append({
            'name': row['name'],
            'artist': row['artist_names'] if isinstance(row['artist_names'], str) else ', '.join(row['artist_names']),
            'genre': row['primary_genre'],
            'popularity': float(row['popularity']),
            'mood': row['mood_label'],
            'track_id': row['track_id'],
            'image': get_track_image(row['track_id']),
            'similarity': float(similarities[idx]),
            'year': int(row['release_year']) if pd.notna(row['release_year']) else 'Unknown'
        })
    
    return jsonify({
        'original': {
            'name': track['name'].values[0],
            'artist': track['artist_names'].values[0],
            'mood': mood,
            'image': get_track_image(track_id)
        },
        'recommendations': recommendations
    })

@app.route('/api/search')
def search_tracks():
    """Search for tracks"""
    query = request.args.get('q', '').lower()
    
    if not query:
        return jsonify([])
    
    # Search in track names
    results = df[df['name'].str.lower().str.contains(query, na=False)].head(10)
    
    search_results = []
    for _, row in results.iterrows():
        search_results.append({
            'name': row['name'],
            'artist': row['artist_names'] if isinstance(row['artist_names'], str) else ', '.join(row['artist_names']),
            'mood': row['mood_label'],
            'image': get_track_image(row['track_id'])
        })
    
    return jsonify(search_results)

@app.route('/api/stats')
def get_stats():
    """Get mood statistics"""
    stats = {}
    for mood in sorted(df['mood_label'].unique()):
        mood_df = df[df['mood_label'] == mood]
        stats[mood] = {
            'count': len(mood_df),
            'avg_popularity': float(mood_df['popularity'].mean()),
            'top_genre': str(mood_df['primary_genre'].mode()[0]) if not mood_df['primary_genre'].mode().empty else 'Unknown',
            'avg_followers': int(mood_df['artist_followers'].mean())
        }
    
    return jsonify(stats)

if __name__ == '__main__':
    print("🎵 Spotify Recommendation Web App Starting...")
    print("📱 Open http://localhost:5000 in your browser")
    app.run(debug=True)
