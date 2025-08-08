import pandas as pd

# Load both CSVs
vol1 = pd.read_csv("nco2015_vol1.csv")  # mapping from Vol I
vol2 = pd.read_csv("nco2015_vol2.csv")  # job descriptions from Vol II

# Merge on 'code'
df = pd.merge(vol2, vol1, on="code", how="left", suffixes=("_desc", "_map"))

# Combine the 'title' and 'description' from the Vol2 side
df['text'] = (df['title_desc'].fillna('') + " " + df['description_desc'].fillna('')).str.replace(r'\s+', ' ', regex=True).str.strip()

# Save cleaned dataset
df.to_csv("nco_clean.csv", index=False)

print(f"✅ Merged dataset saved as nco_clean.csv with {len(df)} rows")
print(df.head())
