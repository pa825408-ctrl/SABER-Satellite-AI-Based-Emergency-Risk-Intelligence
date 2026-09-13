import pandas as pd
from pathlib import Path
import shutil
import random

# ==========================================
# PATHS
# ==========================================

PROJECT = Path(".")

IMAGE_DIR = (
    PROJECT
    / "data"
    / "satellite"
    / "insat3d"
    / "insat3d_ir_cyclone_ds"
)

LABEL_FILE = (
    PROJECT
    / "data"
    / "satellite"
    / "insat3d"
    / "insat_3d_ds - Sheet.csv"
)

OUTPUT_DIR = PROJECT / "data" / "dataset"

TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "validation"
TEST_DIR = OUTPUT_DIR / "test"

METADATA_OUTPUT = (
    PROJECT
    / "data"
    / "processed"
    / "insat_ir_metadata.csv"
)


# ==========================================
# CREATE DIRECTORIES
# ==========================================

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# LOAD LABEL FILE
# ==========================================

print("Loading label file...")

labels = pd.read_csv(LABEL_FILE)

print("Label columns:")
print(labels.columns.tolist())

print("Total label records:", len(labels))


# ==========================================
# FIND ACTUAL IR IMAGES
# ==========================================

print("\nSearching for IR images...")

images = list(IMAGE_DIR.glob("*.jpg"))

print("IR images found:", len(images))


# ==========================================
# CREATE IMAGE LOOKUP
# ==========================================

image_lookup = {
    image.name: image
    for image in images
}


# ==========================================
# MATCH IMAGES WITH LABELS
# ==========================================

records = []

for _, row in labels.iterrows():

    image_name = str(row["img_name"])

    # Only use images that actually exist
    if image_name in image_lookup:

        image_path = image_lookup[image_name]

        records.append({
            "image_name": image_name,
            "image_path": str(image_path),
            "intensity_knots": float(row["label"])
        })


dataset = pd.DataFrame(records)


# ==========================================
# REMOVE DUPLICATES
# ==========================================

dataset = dataset.drop_duplicates(
    subset=["image_name"]
)


# ==========================================
# SHOW DATASET
# ==========================================

print("\n================================")
print("DATASET CREATED")
print("================================")

print("Images matched:", len(dataset))

print("\nFirst records:")
print(dataset.head(10).to_string(index=False))

print("\nIntensity range:")
print(
    dataset["intensity_knots"].min(),
    "to",
    dataset["intensity_knots"].max(),
    "knots"
)


# ==========================================
# SAVE METADATA
# ==========================================

dataset.to_csv(
    METADATA_OUTPUT,
    index=False
)

print(
    "\nMetadata saved to:",
    METADATA_OUTPUT
)


# ==========================================
# TRAIN / VALIDATION / TEST SPLIT
# ==========================================

random.seed(42)

records = dataset.to_dict("records")

random.shuffle(records)

total = len(records)

train_end = int(total * 0.70)
val_end = int(total * 0.85)

train_records = records[:train_end]
val_records = records[train_end:val_end]
test_records = records[val_end:]


# ==========================================
# COPY IMAGES
# ==========================================

def copy_images(records, destination):

    for record in records:

        source = Path(record["image_path"])

        destination_file = (
            destination / record["image_name"]
        )

        shutil.copy2(
            source,
            destination_file
        )


print("\nCopying training images...")

copy_images(
    train_records,
    TRAIN_DIR
)

print("Copying validation images...")

copy_images(
    val_records,
    VAL_DIR
)

print("Copying test images...")

copy_images(
    test_records,
    TEST_DIR
)


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n================================")
print("SPLIT COMPLETE")
print("================================")

print("Training:", len(train_records))
print("Validation:", len(val_records))
print("Testing:", len(test_records))

print("\nDataset location:")
print(OUTPUT_DIR)