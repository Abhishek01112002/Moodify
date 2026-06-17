import pandas as pd
import numpy as np
from pathlib import Path
import faiss
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpotifyFAISSRetriever:
    def __init__(self):
        self.df = None
        self.index = None
        self.feature_columns = None
        self.mean = None
        self.std = None
        self.index_path = Path("src/retrieval/faiss_index.index")
        self.metadata_path = Path("src/retrieval/metadata.pkl")

    def build_index(self):
        """Build FAISS index using real audio features"""
        logger.info("Loading large cleaned dataset...")
        self.df = pd.read_parquet("data/processed/tracks_large_cleaned.parquet")
        
        print(f"✅ Loaded {len(self.df):,} tracks")

        # Audio Features for embedding
        self.feature_columns = [
            "danceability", "energy", "valence", "tempo", "speechiness",
            "acousticness", "instrumentalness", "liveness", "popularity"
        ]
        
        self.feature_columns = [col for col in self.feature_columns if col in self.df.columns]
        
        print(f"Using {len(self.feature_columns)} audio features: {self.feature_columns}")

        # Create feature matrix
        feature_matrix = self.df[self.feature_columns].values.astype(np.float32)

        # Normalize (Critical for cosine similarity)
        self.mean = feature_matrix.mean(axis=0)
        self.std = feature_matrix.std(axis=0) + 1e-8
        feature_matrix = (feature_matrix - self.mean) / self.std

        # Build FAISS Index (Inner Product = Cosine after L2 normalization)
        dimension = feature_matrix.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(feature_matrix)

        logger.info(f"✅ FAISS Index built successfully! Dimension: {dimension}, Tracks: {len(self.df):,}")

        # Save Index + Metadata
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

        metadata = {
            "df": self.df,
            "feature_columns": self.feature_columns,
            "mean": self.mean,
            "std": self.std
        }
        with open(self.metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        logger.info("✅ FAISS Index and metadata saved!")

    def recommend(self, query_track_id: str, top_k: int = 10):
        """Recommend similar tracks based on audio features"""
        if self.df is None:
            self.load_index()

        if query_track_id not in self.df['id'].values:
            print(f"❌ Track ID {query_track_id} not found!")
            return None

        # Get query track features
        query_row = self.df[self.df['id'] == query_track_id].iloc[0]
        query_features = query_row[self.feature_columns].values.astype(np.float32).reshape(1, -1)
        
        # Normalize query
        query_features = (query_features - self.mean) / self.std

        # Search
        distances, indices = self.index.search(query_features, top_k + 1)
        
        # Remove self
        rec_indices = indices[0][1:]
        rec_distances = distances[0][1:]

        recommendations = self.df.iloc[rec_indices].copy()
        recommendations['similarity_score'] = rec_distances

        # Hybrid Reranking: Similarity + Popularity
        recommendations['final_score'] = (
            recommendations['similarity_score'] * 0.7 + 
            (recommendations['popularity'] / 100) * 0.3
        )

        recommendations = recommendations.sort_values('final_score', ascending=False)

        return recommendations.head(top_k)[['id', 'name', 'artists', 'popularity', 'similarity_score', 'final_score']]

    def load_index(self):
        """Load saved FAISS index"""
        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "rb") as f:
            metadata = pickle.load(f)
            self.df = metadata["df"]
            self.feature_columns = metadata["feature_columns"]
            self.mean = metadata["mean"]
            self.std = metadata["std"]
        logger.info("✅ FAISS Index loaded successfully!")

# ==================== QUICK TEST ====================
if __name__ == "__main__":
    retriever = SpotifyFAISSRetriever()
    retriever.build_index()
    
    # Test with a random popular track
    sample_id = retriever.df.sample(1)['id'].iloc[0]
    print(f"\n🔍 Testing recommendation for track_id: {sample_id}")
    
    recs = retriever.recommend(sample_id, top_k=8)
    if recs is not None:
        print("\n✅ Top Recommendations:")
        print(recs[['name', 'artists', 'popularity', 'final_score']].round(3))
