# Legacy Code Archive

This folder contains archived implementations that have been superseded by more modern approaches.

## Flask Demo (`app_flask_legacy.py`)

**Status:** Archived — Use Streamlit instead  
**Reason:** The Flask implementation has been replaced by a more polished Streamlit interface.

### Why the change?

- **Streamlit** is purpose-built for ML demos with live interactivity
- **Streamlit** requires less boilerplate (no templates, routing, JSON endpoints)
- **Streamlit** handles session state and caching automatically
- **Streamlit** integrates better with modern data viz (Plotly, etc.)
- The new implementation at `app/main.py` includes:
  - Hybrid retrieval (FAISS + two-tower + TF-IDF + fuzzy search)
  - Natural language vibe search
  - Recommendation explanations
  - Better error handling
  - Type hints and logging

### How to run the official demo

```bash
poetry run streamlit run app/main.py
```

### If you need the Flask version

You can still reference or run this file as:

```bash
python legacy/app_flask_legacy.py
```

However, we recommend using the Streamlit demo for production and demonstrations.

---

**Archived by:** Portfolio improvement pass (June 2026)  
**Keep for:** Historical reference, learning resources
