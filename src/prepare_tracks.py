import pandas as pd
from pathlib import Path

# --------------------------------
# INPUT AND OUTPUT
# --------------------------------

INPUT = Path("../data/raw/ibtracs.NI.list.v04r01.csv")
OUTPUT = Path("../data/processed/cyclone_tracks.csv")

# --------------------------------
# LOAD DATA
# --------------------------------

print("Loading IBTrACS North Indian Ocean data...")

df = pd.read_csv(
    INPUT,
    skiprows=[1]
)

print("Data loaded successfully!")
print("Total rows:", len(df))

# --------------------------------
# SHOW COLUMNS
# --------------------------------

print("\nAvailable columns:")
print(df.columns.tolist())

# --------------------------------
# CONVERT DATA TYPES
# --------------------------------

df["ISO_TIME"] = pd.to_datetime(
    df["ISO_TIME"],
    errors="coerce"
)

df["LAT"] = pd.to_numeric(
    df["LAT"],
    errors="coerce"
)

df["LON"] = pd.to_numeric(
    df["LON"],
    errors="coerce"
)

df["USA_WIND"] = pd.to_numeric(
    df["USA_WIND"],
    errors="coerce"
)

df["USA_PRES"] = pd.to_numeric(
    df["USA_PRES"],
    errors="coerce"
)

# --------------------------------
# REMOVE INVALID LOCATION/TIME
# --------------------------------

df = df.dropna(
    subset=["ISO_TIME", "LAT", "LON"]
)

# --------------------------------
# SELECT IMPORTANT COLUMNS
# --------------------------------

columns = [
    "SID",
    "SEASON",
    "NUMBER",
    "BASIN",
    "NAME",
    "ISO_TIME",
    "LAT",
    "LON",
    "USA_WIND",
    "USA_PRES"
]

columns = [
    column
    for column in columns
    if column in df.columns
]

clean = df[columns].copy()

# --------------------------------
# SORT
# --------------------------------

clean = clean.sort_values(
    ["SID", "ISO_TIME"]
)


# SAVE

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

clean.to_csv(
    OUTPUT,
    index=False
)

print("\n--------------------------------")
print("SUCCESS!")
print("--------------------------------")

print("Output file:")
print(OUTPUT)

print("\nFirst 10 rows:")
print(clean.head(10).to_string(index=False))