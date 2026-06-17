# get_artists_info.py
from spotipy import Spotify
from spotipy.exceptions import SpotifyException
from tqdm import tqdm
import time

def get_artists_info(sp: Spotify, artist_ids: list, max_retries=3):
    """
    Fetches artist information including genres, followers, and popularity.
    Batches requests in chunks of 50 ids (API limit).
    Handles rate limiting and retries.
    
    Args:
        sp: Authenticated Spotify client
        artist_ids: List of Spotify artist IDs
        max_retries: Maximum number of retries for failed requests
        
    Returns:
        List of artist info dictionaries
    """
    all_info = []
    BATCH = 50
    
    # Filter out None values and get unique artist IDs
    artist_ids = list(set([aid for aid in artist_ids if aid]))
    
    for i in tqdm(range(0, len(artist_ids), BATCH), desc="Fetching artist info"):
        batch = artist_ids[i:i+BATCH]
        retry_count = 0
        success = False
        
        while not success and retry_count < max_retries:
            try:
                res = sp.artists(batch)  # returns artists list
                all_info.extend(res['artists'])
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
                    print(f"Error fetching artist info for batch {i//BATCH + 1}: {e}")
                    # Add empty dicts for this batch
                    all_info.extend([{}] * len(batch))
                    success = True
            except Exception as e:
                print(f"Unexpected error fetching artist info for batch {i//BATCH + 1}: {e}")
                # Add empty dicts for this batch
                all_info.extend([{}] * len(batch))
                success = True
        
        # Small delay to avoid hitting rate limits
        time.sleep(0.1)
    
    return all_info

