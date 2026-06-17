# Deployment Guide

The official demo app is the Streamlit app at `app/main.py`.

## Streamlit Cloud

1. Push the repository to GitHub.
2. Create a new Streamlit Cloud app.
3. Set the app entry point to:

```text
app/main.py
```

4. Add these secrets in Streamlit Cloud if you want Spotify previews and links:

```toml
SPOTIFY_CLIENT_ID = "your_client_id"
SPOTIFY_CLIENT_SECRET = "your_client_secret"
```

5. Make sure the FAISS metadata files needed by the demo are available:

```text
src/retrieval/faiss_index.index
src/retrieval/metadata.pkl
```

For very large artifacts, use a release asset, cloud storage, DVC remote, or
Hugging Face dataset instead of committing them directly.

## Hugging Face Spaces

Create a new Space:

- SDK: Docker
- Python: 3.11

The included `Dockerfile` starts:

```bash
streamlit run app/main.py --server.port=8501 --server.address=0.0.0.0
```

Add Spotify credentials as Space secrets if needed.

## Local Docker

```bash
docker build -t moodify .
docker run --env-file .env -p 8501:8501 moodify
```

Open:

```text
http://localhost:8501
```

## Production Notes

- Do not commit `.env`, `Auth.env`, or API keys.
- Rotate Spotify credentials if they were ever exposed.
- Keep model/index artifacts versioned with DVC, object storage, or release
  assets when they become too large for GitHub.
- Re-run `scripts/evaluate_recommender.py` after every major ranking change.
