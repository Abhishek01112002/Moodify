# auth_helper.py
import os
from dotenv import load_dotenv
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

def get_spotify_client(scope="playlist-read-private,user-top-read,user-library-read"):
    """
    Creates and returns an authenticated Spotify client.
    
    Args:
        scope: Space-separated string of Spotify scopes
        
    Returns:
        Authenticated Spotify client instance
        
    Environment variables required:
        SPOTIPY_CLIENT_ID: Your Spotify app client ID
        SPOTIPY_CLIENT_SECRET: Your Spotify app client secret
        SPOTIPY_REDIRECT_URI: Your redirect URI (e.g., http://127.0.0.1:8888/callback)
    """
    sp_oauth = SpotifyOAuth(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
        scope=scope,
        cache_path=".cache"
    )
    sp = Spotify(auth_manager=sp_oauth)
    return sp

# Usage:
# sp = get_spotify_client()
# me = sp.current_user()
# print(me['display_name'], me['id'])

