from pathlib import Path

from polymer_hsp.io import read_dataset
from polymer_hsp.preprocessing import remove_empty_features

DATASET = Path("data/raw/increased_MW_data_all.xlsx")

df = read_dataset(DATASET, sheet_name="DeltaD")
df = remove_empty_features(df)

print(df.shape)
print(df.head())