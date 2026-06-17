"""
Exploratory Data Analysis (EDA) for Spotify dataset.
Creates visualizations and statistical analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def plot_valence_distribution(df, mood_col='mood_label', save_path=None):
    """
    Plot distribution of valence by mood (happy vs sad patterns).
    
    Args:
        df: DataFrame with valence and mood_label columns
        mood_col: Name of mood column
        save_path: Path to save figure (optional)
    """
    if 'valence' not in df.columns or mood_col not in df.columns:
        print("Missing required columns for valence distribution plot.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall valence distribution
    axes[0].hist(df['valence'].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(df['valence'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {df["valence"].mean():.2f}')
    axes[0].set_xlabel('Valence (0 = Sad, 1 = Happy)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Overall Valence Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Valence by mood
    mood_order = ['sad', 'romantic', 'chill', 'neutral', 'party', 'old_melody']
    available_moods = [m for m in mood_order if m in df[mood_col].values]
    
    if available_moods:
        df_mood = df[df[mood_col].isin(available_moods)]
        sns.boxplot(data=df_mood, x=mood_col, y='valence', ax=axes[1], order=available_moods)
        axes[1].set_title('Valence Distribution by Mood')
        axes[1].set_xlabel('Mood')
        axes[1].set_ylabel('Valence')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved valence distribution plot to {save_path}")
    else:
        plt.show()

def plot_correlation_heatmap(df, features=None, save_path=None):
    """
    Plot correlation heatmap between audio features.
    
    Args:
        df: DataFrame with audio features
        features: List of features to include (default: key audio features)
        save_path: Path to save figure (optional)
    """
    if features is None:
        features = ['energy', 'valence', 'danceability', 'tempo', 'acousticness', 
                   'loudness', 'speechiness', 'instrumentalness', 'liveness']
    
    # Filter to available features
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) < 2:
        print("Not enough features available for correlation heatmap.")
        return
    
    # Calculate correlation
    corr_matrix = df[available_features].corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap: Audio Features', fontsize=16, pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved correlation heatmap to {save_path}")
    else:
        plt.show()

def plot_mood_relationships(df, mood_col='mood_label', save_path=None):
    """
    Visualize relationships between moods and audio features.
    
    Args:
        df: DataFrame with mood labels and audio features
        mood_col: Name of mood column
        save_path: Path to save figure (optional)
    """
    if mood_col not in df.columns:
        print(f"Column '{mood_col}' not found.")
        return
    
    # Key features for mood analysis
    features = ['valence', 'energy', 'danceability', 'tempo', 'acousticness']
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) < 2:
        print("Not enough features available for mood relationship plot.")
        return
    
    # Create subplots
    n_features = len(available_features)
    n_cols = 2
    n_rows = (n_features + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    mood_order = ['sad', 'romantic', 'chill', 'neutral', 'party', 'old_melody']
    available_moods = [m for m in mood_order if m in df[mood_col].values]
    
    for idx, feature in enumerate(available_features):
        if idx < len(axes):
            sns.boxplot(data=df, x=mood_col, y=feature, ax=axes[idx], order=available_moods)
            axes[idx].set_title(f'{feature.capitalize()} by Mood')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(available_features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved mood relationships plot to {save_path}")
    else:
        plt.show()

def plot_cluster_visualization(df, mood_col='mood_label', method='pca', n_components=2, save_path=None):
    """
    Create 2D cluster visualization using PCA or t-SNE.
    
    Args:
        df: DataFrame with features
        mood_col: Name of mood column for coloring
        method: 'pca' or 'tsne'
        n_components: Number of components (2 for 2D visualization)
        save_path: Path to save figure (optional)
    """
    # Select numerical features for dimensionality reduction
    feature_cols = ['valence', 'energy', 'danceability', 'tempo', 'acousticness',
                   'loudness', 'speechiness', 'instrumentalness', 'liveness']
    available_features = [f for f in feature_cols if f in df.columns]
    
    if len(available_features) < 2:
        print("Not enough features for cluster visualization.")
        return
    
    # Prepare data
    X = df[available_features].dropna()
    
    if len(X) == 0:
        print("No valid data points after dropping NaN values.")
        return
    
    # Get corresponding mood labels
    X_indices = X.index
    y = df.loc[X_indices, mood_col] if mood_col in df.columns else None
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply dimensionality reduction
    print(f"Applying {method.upper()}...")
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
        X_reduced = reducer.fit_transform(X_scaled)
        explained_var = reducer.explained_variance_ratio_
        print(f"  Explained variance: {explained_var[0]:.2%} (PC1), {explained_var[1]:.2%} (PC2)")
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
        X_reduced = reducer.fit_transform(X_scaled)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    if y is not None:
        # Color by mood
        unique_moods = y.unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_moods)))
        
        for mood, color in zip(unique_moods, colors):
            mask = y == mood
            plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                       label=mood, alpha=0.6, s=50, c=[color])
        
        plt.legend(title='Mood', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.6, s=50)
    
    method_name = 'PCA' if method.lower() == 'pca' else 't-SNE'
    plt.title(f'{method_name} Visualization of Songs by Mood', fontsize=16)
    plt.xlabel(f'{method_name} Component 1')
    plt.ylabel(f'{method_name} Component 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {method_name} visualization to {save_path}")
    else:
        plt.show()

def generate_eda_report(df, output_dir='eda_output', mood_col='mood_label'):
    """
    Generate complete EDA report with all visualizations.
    
    Args:
        df: DataFrame to analyze
        output_dir: Directory to save plots
        mood_col: Name of mood column
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    
    # Basic statistics
    print("\n" + "=" * 60)
    print("BASIC STATISTICS")
    print("=" * 60)
    if mood_col in df.columns:
        print(f"\nMood Distribution:")
        print(df[mood_col].value_counts())
    
    # Audio features summary
    audio_features = ['valence', 'energy', 'danceability', 'tempo', 'acousticness']
    available_features = [f for f in audio_features if f in df.columns]
    if available_features:
        print(f"\nAudio Features Summary:")
        print(df[available_features].describe())
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # 1. Valence distribution
    print("\n[1/4] Plotting valence distribution...")
    plot_valence_distribution(df, mood_col, 
                             save_path=os.path.join(output_dir, 'valence_distribution.png'))
    
    # 2. Correlation heatmap
    print("\n[2/4] Plotting correlation heatmap...")
    plot_correlation_heatmap(df, save_path=os.path.join(output_dir, 'correlation_heatmap.png'))
    
    # 3. Mood relationships
    print("\n[3/4] Plotting mood relationships...")
    plot_mood_relationships(df, mood_col, 
                           save_path=os.path.join(output_dir, 'mood_relationships.png'))
    
    # 4. Cluster visualization (PCA)
    print("\n[4/4] Creating PCA visualization...")
    plot_cluster_visualization(df, mood_col, method='pca', 
                              save_path=os.path.join(output_dir, 'pca_visualization.png'))
    
    # Optional: t-SNE (slower, comment out if needed)
    # print("\n[5/5] Creating t-SNE visualization...")
    # plot_cluster_visualization(df, mood_col, method='tsne', 
    #                           save_path=os.path.join(output_dir, 'tsne_visualization.png'))
    
    print("\n" + "=" * 60)
    print(f"EDA complete! All plots saved to '{output_dir}' directory")
    print("=" * 60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else 'eda_output'
        
        print(f"Loading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        generate_eda_report(df, output_dir=output_dir)
    else:
        print("Usage: python eda.py <input_file> [output_dir]")

