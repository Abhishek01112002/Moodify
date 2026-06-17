# mood_labeling.py
"""
Mood labeling functions using rule-based approach on audio features.
"""

def label_mood_by_rules(row, artist_genres=None):
    """
    Labels a song with mood based on audio features and metadata.
    
    Args:
        row: DataFrame row or dict with audio features
        artist_genres: List of genre strings for the artist (optional)
        
    Returns:
        String mood label: 'party', 'sad', 'romantic', 'chill', 'old_melody', or 'neutral'
    """
    # Extract features (handle missing values)
    valence = row.get('valence', 0.5)
    energy = row.get('energy', 0.5)
    danceability = row.get('danceability', 0.5)
    tempo = row.get('tempo', 100)
    acousticness = row.get('acousticness', 0.5)
    release_date = row.get('release_date', '')
    
    # Extract year from release_date (format: YYYY-MM-DD or YYYY)
    release_year = None
    if release_date:
        try:
            release_year = int(release_date.split('-')[0])
        except (ValueError, AttributeError):
            pass
    
    # Rule 1: Party
    if energy > 0.7 and danceability > 0.7 and tempo > 110:
        return 'party'
    
    # Rule 2: Sad
    if valence < 0.35 and acousticness > 0.5:
        return 'sad'
    
    # Rule 3: Romantic
    if 0.4 <= valence <= 0.7 and acousticness > 0.4 and tempo < 100:
        return 'romantic'
    
    # Rule 4: Chill (relaxed, low energy, medium-high valence)
    if energy < 0.5 and valence > 0.5 and danceability < 0.6 and tempo < 120:
        return 'chill'
    
    # Rule 5: Old Melody
    if release_year and release_year < 1995:
        return 'old_melody'
    
    # Check artist genres for old_melody
    if artist_genres:
        genre_str = ' '.join(artist_genres).lower()
        if any(keyword in genre_str for keyword in ['oldies', 'classic', 'vintage', 'retro']):
            return 'old_melody'
    
    # Default: neutral
    return 'neutral'

def label_from_playlist(playlist_mood):
    """
    Returns a function that labels all tracks with the given playlist mood.
    Useful for playlist-based weak labeling.
    
    Args:
        playlist_mood: Mood label to assign (e.g., 'sad', 'party', 'romantic')
        
    Returns:
        Function that takes a row and returns the playlist mood
    """
    def label_func(row):
        return playlist_mood
    return label_func

