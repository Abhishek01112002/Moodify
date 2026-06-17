import pandas as pd
df = pd.read_parquet('data/processed/tracks_cleaned.parquet')
print('Columns:', df.columns.tolist())
print('\nSample row:')
print(df.iloc[0])
print('\nShape:', df.shape)
