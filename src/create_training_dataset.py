import pandas as pd
from pathlib import Path

# ==========================================
# FILE PATHS
# ==========================================

INPUT = Path("../data/processed/cyclone_tracks.csv")

OUTPUT = Path("../data/processed/training_metadata.csv")


# ==========================================
# LOAD CLEAN CYCLONE DATA
# ==========================================

print("Loading cyclone tracks...")

df = pd.read_csv(INPUT)

print("Loaded successfully!")
print("Total records:", len(df))


# ==========================================
# CONVERT DATA TYPES
# ==========================================

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


# ==========================================
# REMOVE INVALID RECORDS
# ==========================================

df = df.dropna(
    subset=[
        "ISO_TIME",
        "LAT",
        "LON"
    ]

)
# KEEP MODERN CYCLONES
# ==========================================

# Satellite data is much more useful for
# modern cyclone cases.

df = df[
    (df["SEASON"] >= 1980)
    & (df["SEASON"] <= 2025)
]


# ==========================================
# REMOVE UNNAMED SYSTEMS
# ==========================================

df["NAME"] = df["NAME"].astype(str).str.strip()

df = df[
    (df["NAME"] != "")
    & (df["NAME"].str.upper() != "UNNAMED")
]

# ==========================================
# CREATE IMAGE ID
# ==========================================

df = df.reset_index(drop=True)

df["image_id"] = [
    f"{i:06d}"
    for i in range(1, len(df) + 1)
]


# ==========================================
# CYCLONE LABEL
# ==========================================

# Every record in IBTrACS NI is a
# known tropical-cyclone track observation.

df["cyclone"] = 1


# ==========================================
# SELECT IMPORTANT INFORMATION
# ==========================================

metadata = df[
    [
        "image_id",
        "cyclone",
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
].copy()


# ==========================================
# ADD IMAGE PATH COLUMN
# ==========================================

# Satellite images will be added later.

metadata["image_path"] = ""


# ==========================================
# SORT BY STORM AND TIME
# ==========================================

metadata = metadata.sort_values(
    [
        "SID",
        "ISO_TIME"
    ]
)


# ==========================================
# SAVE DATASET
# ==========================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

metadata.to_csv(
    OUTPUT,
    index=False
)


# ==========================================
# SHOW RESULT
# ==========================================

print()
print("==========================================")
print("TRAINING DATASET CREATED")
print("==========================================")

print("File:")
print(OUTPUT)

print()
print("Number of records:")
print(len(metadata))

print()
print("Number of cyclones:")
print(metadata["SID"].nunique())

print()
print("Cyclone names:")
print(
    metadata["NAME"]
    .dropna()
    .unique()
)

print()
print("First 10 records:")
print(
    metadata.head(10).to_string(index=False)
)

print()
print("SUCCESS!")