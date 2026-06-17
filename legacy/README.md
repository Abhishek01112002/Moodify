# Legacy Code Archive

This folder contains archived implementations that have been fully replaced.

## Flask Demo (`app_flask_legacy.py`)

**Status:** Deprecated & Removed — Replaced by FastAPI in `app_fastapi.py`  
**Reason:** The Flask implementation was legacy and has been replaced by a modern, high-fidelity FastAPI Single Page Application.

### Current implementations:
- **Streamlit App (`app/main.py`)**: Official demo dashboard.
- **FastAPI Web App (`app_fastapi.py`)**: High-fidelity modern Single Page Application (SPA) served by FastAPI.

### How to run the Streamlit app

```bash
poetry run streamlit run app/main.py
```

### How to run the FastAPI web app

```bash
poetry run python app_fastapi.py
```

Then open `http://localhost:8000` in your browser.
