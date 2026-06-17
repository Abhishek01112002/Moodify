# fetch_playlists_and_tracks.py
from spotipy import Spotify
from spotipy.exceptions import SpotifyException
from tqdm import tqdm
import time

def get_all_playlist_tracks(sp: Spotify, playlist_id: str, max_retries=3):
    """
    Returns a list of track objects (Spotify API track items) from a playlist.
    Handles pagination and rate limiting.
    
    Args:
        sp: Authenticated Spotify client
        playlist_id: Spotify playlist ID
        max_retries: Maximum number of retries for failed requests
        
    Returns:
        List of track objects from the playlist
    """
    tracks = []
    results = None
    retry_count = 0
    
    try:
        results = sp.playlist_items(
            playlist_id, 
            additional_types=['track'], 
            fields="items.track.id,items.track.name,items.track.artists,items.track.popularity,items.track.album,next"
        )
    except SpotifyException as e:
        if e.http_status == 429:  # Rate limit
            retry_after = 5
            if e.headers and e.headers.get('Retry-After'):
                retry_after = int(e.headers.get('Retry-After'))
            print(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after + 1)
            if retry_count < max_retries:
                retry_count += 1
                return get_all_playlist_tracks(sp, playlist_id, max_retries)
        raise
    
    while results:
        for item in results['items']:
            track = item.get('track')
            if not track or not track.get('id'): 
                continue
            tracks.append(track)
        
        if results.get('next'):
            try:
                results = sp.next(results)
            except SpotifyException as e:
                if e.http_status == 429:  # Rate limit
                    retry_after = 5
                    if e.headers and e.headers.get('Retry-After'):
                        retry_after = int(e.headers.get('Retry-After'))
                    print(f"Rate limited. Waiting {retry_after} seconds...")
                    time.sleep(retry_after + 1)
                    # Retry the same request
                    continue
                else:
                    print(f"Error fetching next page: {e}")
                    break
            except Exception as e:
                print(f"Unexpected error fetching next page: {e}")
                break
        else:
            results = None
    
    return tracks

# Example usage:
# sp = get_spotify_client()
# playlist_id = "37i9dQZF1DXcBWIGoYBM5M"  # example
# tracks = get_all_playlist_tracks(sp, playlist_id)
# print(len(tracks))

