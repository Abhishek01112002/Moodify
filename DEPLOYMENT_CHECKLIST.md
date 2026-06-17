# Deployment Checklist — Moodify on Streamlit Cloud

This guide walks you through deploying the Moodify app to **Streamlit Cloud** (free tier) so you can share a live link on your resume and GitHub profile.

---

## Step 1: Push to GitHub (Required First)

```bash
# Create a new repo on GitHub first (https://github.com/new)
# Name it: moodify (or whatever you prefer)

git remote add origin https://github.com/YOUR_USERNAME/moodify.git
git branch -M main
git push -u origin main
```

After pushing, **update the badge** in `README.md` line 3:
```markdown
![CI](https://github.com/YOUR_USERNAME/moodify/actions/workflows/ci.yml/badge.svg)
```

Commit and push the badge update:
```bash
git add README.md
git commit -m "docs: update GitHub Actions badge"
git push
```

---

## Step 2: Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **"New app"**
3. Under **"Deploy an app"**, select:
   - **Repository:** `YOUR_USERNAME/moodify`
   - **Branch:** `main`
   - **Main file path:** `app/main.py`
4. Click **"Deploy"**

The app will build automatically (takes ~3–5 minutes).

---

## Step 3: Add Spotify Secrets (Optional but Recommended)

Without Spotify credentials, the app works but **won't show album artwork or audio previews**.

1. In your deployed app on Streamlit Cloud, click the **"⋮"** (menu) → **"Settings"**
2. Go to the **"Secrets"** tab
3. Add the following:

```toml
SPOTIFY_CLIENT_ID = "your_client_id"
SPOTIFY_CLIENT_SECRET = "your_client_secret"
```

> Get these from: [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)

4. Click **"Save"** and the app will restart with full features

---

## Step 4: Add the Live Link to Your README

Once deployed, copy your Streamlit Cloud URL (e.g., `https://moodify-username.streamlit.app`) and add it to the top of your README:

```markdown
# Moodify: Hybrid Spotify Music Recommendation Engine

[![CI](https://github.com/YOUR_USERNAME/moodify/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/moodify/actions)
**[Live Demo →](https://moodify-username.streamlit.app)**
```

Commit and push:
```bash
git add README.md
git commit -m "docs: add live Streamlit Cloud demo link"
git push
```

---

## Step 5: Add to Your Resume / LinkedIn

Update your resume bullet with the live link:

> Built **Moodify**, an end-to-end hybrid music recommender with **FAISS vector search**, **TF-IDF/fuzzy text retrieval**, **self-supervised contrastive learning** (SimCLR-style item tower), and a **profile-aware two-tower model**. Live demo at **[moodify-username.streamlit.app](https://moodify-username.streamlit.app)** with 89K+ tracks, explainable recommendations, and Spotify previews. 19 unit tests, GitHub Actions CI, MLflow experiment tracking.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `FileNotFoundError: faiss_index.index` | The FAISS index files are committed to the repo. Make sure `.gitignore` exceptions are correct (see `.gitignore` lines 31–35). |
| Build takes >5 minutes | First deployment downloads all dependencies. Subsequent deployments use cache. |
| No album artwork | Add Spotify credentials in Streamlit Cloud Secrets. |
| `ModuleNotFoundError: app.main` | Ensure `app/__init__.py` exists in the repo. |
| CI badge is grey | Push to GitHub first. The badge only works after the first CI run. |

---

## Alternative: Docker / Hugging Face Spaces

If you prefer Docker-based deployment (e.g., Hugging Face Spaces):

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Docker** as the SDK
3. Set the container port to **8501**
4. The included `Dockerfile` handles the rest:

```dockerfile
FROM python:3.11-slim
# ... (see Dockerfile in repo)
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

**Estimated total time to deploy:** 5–10 minutes (after GitHub push)
