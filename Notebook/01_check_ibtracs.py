import pandas as pd

FILE = "../data/raw/ibtracs.NI.list.v04r01.csv"

df = pd.read_csv(
    FILE,
    skiprows=[1]
)

print("Rows:", len(df))
print("Columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print(df["BASIN"].value_counts())
north_indian = df[df["BASIN"] == "NI"].copy()

print("North Indian Ocean records:", len(north_indian))
print(north_indian["NAME"].unique())
north_indian.to_csv(
    "../data/processed/north_indian_ocean_cyclones.csv",
    index=False
)




