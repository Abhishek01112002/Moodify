"""
Data preprocessing module for Spotify dataset.
Handles cleaning, normalization, encoding, and feature engineering.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def remove_duplicates(df, subset='track_id'):
    """
    Remove duplicate tracks based on track_id.
    
    Args:
        df: DataFrame with track data
        subset: Column(s) to use for duplicate detection
        
    Returns:
        DataFrame with duplicates removed
    """
    initial_count = len(df)
    df_clean = df.drop_duplicates(subset=subset, keep='first')
    removed = initial_count - len(df_clean)
    print(f"Removed {removed} duplicate tracks. Remaining: {len(df_clean)}")
    return df_clean

def handle_missing_values(df, strategy='median'):
    """
    Handle missing values in numerical columns.
    
    Args:
        df: DataFrame with potential missing values
        strategy: Imputation strategy ('mean', 'median', 'most_frequent', 'constant')
        
    Returns:
        DataFrame with missing values imputed
    """
    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove track_id and other ID columns from imputation
    id_cols = ['track_id', 'artist_id'] if 'artist_id' in df.columns else ['track_id']
    numerical_cols = [col for col in numerical_cols if col not in id_cols]
    
    # Count missing values
    missing_counts = df[numerical_cols].isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    
    if len(missing_cols) > 0:
        print(f"\nMissing values found in {len(missing_cols)} columns:")
        for col, count in missing_cols.items():
            print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
        
        # Impute missing values
        imputer = SimpleImputer(strategy=strategy)
        df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
        print(f"\nImputed missing values using '{strategy}' strategy.")
    else:
        print("No missing values found in numerical columns.")
    
    # Handle categorical/text columns
    text_cols = ['name', 'album', 'artist_names', 'primary_genre', 'all_genres']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    
    return df

def normalize_features(df, method='standard', exclude_cols=None):
    """
    Normalize numerical features using StandardScaler or MinMaxScaler.
    
    Args:
        df: DataFrame with numerical features
        method: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
        exclude_cols: List of columns to exclude from normalization
        
    Returns:
        DataFrame with normalized features and fitted scaler
    """
    if exclude_cols is None:
        exclude_cols = ['track_id', 'artist_id', 'mood_label', 'popularity', 
                       'artist_popularity', 'artist_followers', 'release_date']
    
    # Identify numerical columns to normalize
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    normalize_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    if not normalize_cols:
        print("No columns to normalize.")
        return df, None
    
    print(f"\nNormalizing {len(normalize_cols)} columns using {method} scaling:")
    print(f"  Columns: {', '.join(normalize_cols[:5])}{'...' if len(normalize_cols) > 5 else ''}")
    
    # Create scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'standard' or 'minmax'")
    
    # Normalize features
    df_normalized = df.copy()
    df_normalized[normalize_cols] = scaler.fit_transform(df[normalize_cols])
    
    return df_normalized, scaler

def encode_genres_multi_hot(df, genre_col='all_genres', max_genres=20):
    """
    Create multi-hot encoding for genres.
    
    Args:
        df: DataFrame with genre information
        genre_col: Column containing genres (semicolon-separated)
        max_genres: Maximum number of top genres to encode
        
    Returns:
        DataFrame with multi-hot encoded genre columns
    """
    if genre_col not in df.columns:
        print(f"Column '{genre_col}' not found. Skipping genre encoding.")
        return df
    
    print(f"\nEncoding genres from '{genre_col}' column...")
    
    # Extract all unique genres
    all_genres = []
    for genres_str in df[genre_col].dropna():
        if genres_str and genres_str != 'Unknown':
            genres = [g.strip() for g in str(genres_str).split(';')]
            all_genres.extend(genres)
    
    # Get top genres by frequency
    genre_counts = pd.Series(all_genres).value_counts()
    top_genres = genre_counts.head(max_genres).index.tolist()
    
    print(f"  Found {len(set(all_genres))} unique genres")
    print(f"  Encoding top {len(top_genres)} genres")
    
    # Create multi-hot encoding
    df_encoded = df.copy()
    for genre in top_genres:
        df_encoded[f'genre_{genre.lower().replace(" ", "_")}'] = df_encoded[genre_col].apply(
            lambda x: 1 if x and genre.lower() in str(x).lower() else 0
        )
    
    return df_encoded

def create_mood_label_column(df, mood_col='mood_label'):
    """
    Ensure mood_label column exists and map values to standard categories.
    
    Args:
        df: DataFrame
        mood_col: Name of mood column
        
    Returns:
        DataFrame with standardized mood labels
    """
    df_processed = df.copy()
    
    # If mood column doesn't exist, create it
    if mood_col not in df_processed.columns:
        df_processed[mood_col] = 'neutral'
        print(f"Created '{mood_col}' column with default value 'neutral'")
    
    # Standard mood categories
    valid_moods = ['sad', 'romantic', 'party', 'chill', 'old_melody', 'neutral']
    
    # Map any variations to standard categories
    mood_mapping = {
        'sad': 'sad',
        'romantic': 'romantic',
        'party': 'party',
        'chill': 'chill',
        'old_melody': 'old_melody',
        'old_melody': 'old_melody',
        'neutral': 'neutral',
        None: 'neutral',
        np.nan: 'neutral'
    }
    
    # Apply mapping
    df_processed[mood_col] = df_processed[mood_col].map(
        lambda x: mood_mapping.get(str(x).lower() if pd.notna(x) else None, 'neutral')
    )
    
    # Print distribution
    print(f"\nMood label distribution:")
    mood_counts = df_processed[mood_col].value_counts()
    for mood, count in mood_counts.items():
        print(f"  {mood}: {count} ({count/len(df_processed)*100:.1f}%)")
    
    return df_processed

def extract_release_year(df, date_col='release_date'):
    """
    Extract release year from release_date column.
    
    Args:
        df: DataFrame with release_date column
        date_col: Name of date column
        
    Returns:
        DataFrame with release_year column added
    """
    df_processed = df.copy()
    
    if date_col not in df_processed.columns:
        print(f"Column '{date_col}' not found. Skipping year extraction.")
        return df_processed
    
    def extract_year(date_str):
        if pd.isna(date_str) or date_str == '':
            return None
        try:
            year_str = str(date_str).split('-')[0]
            year = int(year_str)
            return year if 1900 <= year <= 2025 else None
        except (ValueError, AttributeError):
            return None
    
    df_processed['release_year'] = df_processed[date_col].apply(extract_year)
    
    # Fill missing years with median
    if df_processed['release_year'].notna().any():
        median_year = df_processed['release_year'].median()
        df_processed['release_year'] = df_processed['release_year'].fillna(median_year)
        print(f"Extracted release years. Median: {int(median_year)}")
    
    return df_processed

def preprocess_dataset(df, normalize_method='standard', encode_genres=True):
    """
    Complete preprocessing pipeline.
    
    Args:
        df: Raw DataFrame from pipeline
        normalize_method: 'standard' or 'minmax'
        encode_genres: Whether to perform multi-hot genre encoding
        
    Returns:
        Preprocessed DataFrame and fitted scaler
    """
    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    print(f"\nInitial dataset shape: {df.shape}")
    
    # Step 1: Remove duplicates
    print("\n[Step 1] Removing duplicates...")
    df = remove_duplicates(df)
    
    # Step 2: Extract release year
    print("\n[Step 2] Extracting release year...")
    df = extract_release_year(df)
    
    # Step 3: Handle missing values
    print("\n[Step 3] Handling missing values...")
    df = handle_missing_values(df, strategy='median')
    
    # Step 4: Create/standardize mood labels
    print("\n[Step 4] Creating mood label column...")
    df = create_mood_label_column(df)
    
    # Step 5: Encode genres
    if encode_genres:
        print("\n[Step 5] Encoding genres (multi-hot)...")
        df = encode_genres_multi_hot(df, max_genres=20)
    
    # Step 6: Normalize features
    print("\n[Step 6] Normalizing numerical features...")
    df, scaler = normalize_features(df, method=normalize_method)
    
    print("\n" + "=" * 60)
    print(f"Preprocessing complete! Final shape: {df.shape}")
    print("=" * 60)
    
    return df, scaler

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "spotify_songs_cleaned.csv"
        
        print(f"Loading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        df_clean, scaler = preprocess_dataset(df)
        
        print(f"\nSaving cleaned data to {output_file}...")
        df_clean.to_csv(output_file, index=False)
        print("Done!")
    else:
        print("Usage: python data_preprocessing.py <input_file> [output_file]")

