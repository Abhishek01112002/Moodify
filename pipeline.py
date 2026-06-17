# pipeline_example.py
"""
Main pipeline script for collecting Spotify data with mood labels.
"""
from auth_helper import get_spotify_client
from fetch_playlists_and_tracks import get_all_playlist_tracks
from get_audio_features import get_audio_features_for_tracks
from get_artists_info import get_artists_info
from mood_labeling import label_mood_by_rules
import pandas as pd

def create_tracks_dataframe(tracks, mood_label=None):
    """
    Creates a DataFrame from track objects.
    
    Args:
        tracks: List of track dictionaries from Spotify API
        mood_label: Optional mood label to assign to all tracks
        
    Returns:
        DataFrame with track metadata
    """
    all_rows = []
    
    for t in tracks:
        if not t.get('id'):
            continue
        
        # Extract artist info
        artists = t.get('artists', [])
        artist_ids = [a['id'] for a in artists if a.get('id')]
        artist_names = [a['name'] for a in artists if a.get('name')]
        
        # Extract album info
        album = t.get('album', {})
        album_name = album.get('name', '')
        release_date = album.get('release_date', '')
        
        row = {
            "track_id": t['id'],
            "name": t.get('name', ''),
            "artist_ids": artist_ids,
            "artist_names": artist_names,
            "popularity": t.get('popularity', 0),
            "album": album_name,
            "release_date": release_date,
        }
        
        # Add mood label if provided
        if mood_label:
            row["mood_label"] = mood_label
        
        all_rows.append(row)
    
    return pd.DataFrame(all_rows)

def merge_audio_features(tracks_df, audio_features):
    """
    Merges audio features with tracks DataFrame.
    
    Args:
        tracks_df: DataFrame with track metadata
        audio_features: List of audio feature dictionaries
        
    Returns:
        Merged DataFrame
    """
    # Filter out None entries and create DataFrame
    valid_features = [f for f in audio_features if f and f.get('id')]
    audio_df = pd.DataFrame(valid_features)
    
    if audio_df.empty:
        return tracks_df
    
    # Rename id column for merging
    audio_df = audio_df.rename(columns={'id': 'track_id'})
    
    # Merge
    df = tracks_df.merge(audio_df, on='track_id', how='left')
    return df

def merge_artist_info(df, artists_info):
    """
    Merges artist information (genres, followers, popularity) with tracks DataFrame.
    
    Args:
        df: DataFrame with tracks and audio features
        artists_info: List of artist info dictionaries
        
    Returns:
        DataFrame with artist info merged
    """
    if not artists_info:
        return df
    
    # Create artist info DataFrame
    artist_df = pd.DataFrame(artists_info)
    
    if artist_df.empty:
        return df
    
    # Extract primary genre and other info
    artist_df['primary_genre'] = artist_df['genres'].apply(
        lambda x: x[0] if x and len(x) > 0 else ''
    )
    artist_df['all_genres'] = artist_df['genres'].apply(
        lambda x: '; '.join(x) if x else ''
    )
    artist_df['artist_followers'] = artist_df['followers'].apply(
        lambda x: x.get('total', 0) if isinstance(x, dict) else 0
    )
    artist_df['artist_popularity'] = artist_df['popularity'] if 'popularity' in artist_df.columns else 0
    
    # Select columns to merge
    artist_merge = artist_df[['id', 'primary_genre', 'all_genres', 'artist_followers', 'artist_popularity']].rename(
        columns={'id': 'artist_id'}
    )
    
    # Expand artist_ids column (handle lists)
    df_expanded = df.explode('artist_ids').rename(columns={'artist_ids': 'artist_id'})
    
    # Merge with artist info
    df_expanded = df_expanded.merge(artist_merge, on='artist_id', how='left')
    
    # Get all columns that should be preserved
    non_artist_cols = [col for col in df.columns if col != 'artist_ids']
    
    # Aggregate back (take first non-null genre for each track)
    agg_dict = {
        'primary_genre': lambda x: x.dropna().iloc[0] if x.dropna().any() else '',
        'all_genres': lambda x: '; '.join(x.dropna().unique()) if x.dropna().any() else '',
        'artist_followers': 'max',  # Max followers among artists
        'artist_popularity': 'max',  # Max popularity among artists
    }
    
    # Add all other columns with 'first' aggregation
    for col in non_artist_cols:
        if col not in agg_dict:
            agg_dict[col] = 'first'
    
    df_agg = df_expanded.groupby('track_id', as_index=False).agg(agg_dict)
    
    return df_agg

def apply_mood_labels(df):
    """
    Applies rule-based mood labeling to tracks that don't have a mood_label.
    
    Args:
        df: DataFrame with tracks and features
        
    Returns:
        DataFrame with mood labels applied
    """
    if 'mood_label' not in df.columns:
        df['mood_label'] = None
    
    # Apply rules to rows without mood labels
    mask = df['mood_label'].isna() | (df['mood_label'] == '')
    
    if mask.any():
        df.loc[mask, 'mood_label'] = df[mask].apply(
            lambda row: label_mood_by_rules(
                row, 
                artist_genres=row.get('all_genres', '').split('; ') if row.get('all_genres') else None
            ),
            axis=1
        )
    
    return df

def run_pipeline(playlists=None, output_file="labeled_spotify_songs.csv", use_playlist_labels=True):
    """
    Main pipeline function to collect Spotify data.
    
    Args:
        playlists: Dict mapping mood labels to playlist IDs
                  Example: {"sad": "PLAYLIST_ID_SAD", "party": "PLAYLIST_ID_PARTY"}
        output_file: Output CSV filename
        use_playlist_labels: If True, use playlist names as mood labels
    """
    print("Initializing Spotify client...")
    sp = get_spotify_client()
    
    # Test connection
    try:
        me = sp.current_user()
        print(f"Connected as: {me.get('display_name', 'Unknown')} ({me.get('id', 'Unknown')})")
    except Exception as e:
        print(f"Error connecting to Spotify: {e}")
        return
    
    # Default playlists if none provided
    if playlists is None:
        print("No playlists provided. Please provide a dict of {mood: playlist_id}")
        return
    
    all_rows = []
    
    # Step 1: Fetch tracks from playlists
    print("\n=== Step 1: Fetching tracks from playlists ===")
    for mood, pid in playlists.items():
        print(f"\nFetching {mood} playlist: {pid}")
        try:
            tracks = get_all_playlist_tracks(sp, pid)
            print(f"  Found {len(tracks)} tracks")
            
            # Create DataFrame for this playlist
            tracks_df = create_tracks_dataframe(tracks, mood_label=mood if use_playlist_labels else None)
            all_rows.append(tracks_df)
        except Exception as e:
            print(f"  Error fetching playlist {pid}: {e}")
            continue
    
    if not all_rows:
        print("No tracks fetched. Exiting.")
        return
    
    # Combine all tracks
    tracks_df = pd.concat(all_rows, ignore_index=True)
    print(f"\nTotal unique tracks: {len(tracks_df)}")
    
    # Remove duplicates
    tracks_df = tracks_df.drop_duplicates(subset='track_id')
    print(f"After removing duplicates: {len(tracks_df)}")
    
    # Step 2: Fetch audio features
    print("\n=== Step 2: Fetching audio features ===")
    track_ids = tracks_df['track_id'].unique().tolist()
    audio_features = get_audio_features_for_tracks(sp, track_ids)
    
    # Step 3: Merge audio features
    print("\n=== Step 3: Merging audio features ===")
    df = merge_audio_features(tracks_df, audio_features)
    
    # Step 4: Fetch artist info
    print("\n=== Step 4: Fetching artist information ===")
    # Get all unique artist IDs
    all_artist_ids = []
    for artist_list in tracks_df['artist_ids']:
        if isinstance(artist_list, list):
            all_artist_ids.extend(artist_list)
        elif artist_list:
            all_artist_ids.append(artist_list)
    
    unique_artist_ids = list(set([aid for aid in all_artist_ids if aid]))
    print(f"Fetching info for {len(unique_artist_ids)} unique artists...")
    
    artists_info = get_artists_info(sp, unique_artist_ids)
    
    # Step 5: Merge artist info
    print("\n=== Step 5: Merging artist information ===")
    df = merge_artist_info(df, artists_info)
    
    # Step 6: Apply mood labels (rule-based for tracks without playlist labels)
    print("\n=== Step 6: Applying mood labels ===")
    df = apply_mood_labels(df)
    
    # Step 7: Save dataset
    print(f"\n=== Step 7: Saving dataset to {output_file} ===")
    df.to_csv(output_file, index=False)
    print(f"Saved dataset with shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns)}")
    
    # Print summary
    print("\n=== Summary ===")
    if 'mood_label' in df.columns:
        print("\nMood distribution:")
        print(df['mood_label'].value_counts())
    
    return df

if __name__ == "__main__":
    # Example usage:
    # Define your playlists here
    playlists = {
        "sad": "YOUR_SAD_PLAYLIST_ID",
        "romantic": "YOUR_ROMANTIC_PLAYLIST_ID",
        "party": "YOUR_PARTY_PLAYLIST_ID",
        "old_melody": "YOUR_OLD_MELODY_PLAYLIST_ID"
    }
    
    # Run pipeline
    # df = run_pipeline(playlists, output_file="labeled_spotify_songs.csv")

