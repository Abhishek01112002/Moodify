"""
Complete workflow for PHASE 2 (Data Preprocessing) and PHASE 3 (EDA).
Run this script after collecting data with the pipeline.
"""
import pandas as pd
import os
import sys

def main():
    print("=" * 70)
    print("SPOTIFY RECOMMENDATION SYSTEM - PHASE 2 & 3")
    print("=" * 70)
    
    # Check if cleaned data exists, otherwise use raw data
    if os.path.exists("labeled_spotify_songs.csv"):
        input_file = "labeled_spotify_songs.csv"
        print(f"\n[Step 1] Found raw data: {input_file}")
    elif os.path.exists("spotify_songs_cleaned.csv"):
        print("\n[Step 1] Using existing cleaned data")
        input_file = "spotify_songs_cleaned.csv"
    else:
        print("\n❌ Error: No data file found!")
        print("   Please run the pipeline first to collect data:")
        print("   python main.py")
        return
    
    # Load data
    print(f"   Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"   Loaded {len(df)} tracks")
    
    # PHASE 2: Data Preprocessing
    print("\n" + "=" * 70)
    print("PHASE 2: DATA PREPROCESSING & CLEANING")
    print("=" * 70)
    
    from data_preprocessing import preprocess_dataset
    
    df_clean, scaler = preprocess_dataset(
        df, 
        normalize_method='standard', 
        encode_genres=True
    )
    
    # Save cleaned data
    output_file = "spotify_songs_cleaned.csv"
    df_clean.to_csv(output_file, index=False)
    print(f"\n✅ Saved cleaned data to {output_file}")
    
    # PHASE 3: Exploratory Data Analysis
    print("\n" + "=" * 70)
    print("PHASE 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    
    from eda import generate_eda_report
    
    output_dir = "eda_output"
    generate_eda_report(df_clean, output_dir=output_dir, mood_col='mood_label')
    
    print("\n" + "=" * 70)
    print("✅ PHASE 2 & 3 COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Cleaned dataset: {output_file}")
    print(f"📈 EDA visualizations: {output_dir}/")
    print("\nNext steps:")
    print("  - Review EDA visualizations in the 'eda_output' directory")
    print("  - Proceed to PHASE 4: Build recommendation engines")

if __name__ == "__main__":
    main()

