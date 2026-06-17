# get_audio_features.py
from spotipy import Spotify
from spotipy.exceptions import SpotifyException
from tqdm import tqdm
import time

def get_audio_features_for_tracks(sp: Spotify, track_ids: list, max_retries=3):
    """
    Accepts list of track ids (strings) and returns list of feature dicts.
    Batches requests in chunks of 100 ids (API limit).
    Handles rate limiting and retries.
    
    Args:
        sp: Authenticated Spotify client
        track_ids: List of Spotify track IDs
        max_retries: Maximum number of retries for failed requests
        
    Returns:
        List of audio feature dictionaries (or None entries for invalid tracks)
    """
    all_features = []
    BATCH = 100
    
    # Filter out None values
    track_ids = [tid for tid in track_ids if tid]
    
    for i in tqdm(range(0, len(track_ids), BATCH), desc="Fetching audio features"):
        batch = track_ids[i:i+BATCH]
        retry_count = 0
        success = False
        
        while not success and retry_count < max_retries:
            try:
                features = sp.audio_features(batch)  # returns list of feature dicts (or None entries)
                all_features.extend(features)
                success = True
            except SpotifyException as e:
                if e.http_status == 429:  # Rate limit
                    retry_after = 5
                    if e.headers and e.headers.get('Retry-After'):
                        retry_after = int(e.headers.get('Retry-After'))
                    print(f"Rate limited. Waiting {retry_after} seconds...")
                    time.sleep(retry_after + 1)
                    retry_count += 1
                else:
                    print(f"Error fetching audio features for batch {i//BATCH + 1}: {e}")
                    # Add None entries for this batch
                    all_features.extend([None] * len(batch))
                    success = True
            except Exception as e:
                print(f"Unexpected error fetching audio features for batch {i//BATCH + 1}: {e}")
                # Add None entries for this batch
                all_features.extend([None] * len(batch))
                success = True
        
        # Small delay to avoid hitting rate limits
        time.sleep(0.1)
    
    return all_features

