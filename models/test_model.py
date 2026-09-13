import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import numpy as np


# ==========================================
# SETTINGS
# ==========================================

PROJECT = Path(".")

METADATA_FILE = (
    PROJECT
    / "data"
    / "processed"
    / "insat_ir_metadata.csv"
)

TEST_DIR = (
    PROJECT
    / "data"
    / "dataset"
    / "test"
)

MODEL_FILE = (
    PROJECT
    / "models"
    / "best_model.pth"
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# LOAD METADATA
# ==========================================

metadata = pd.read_csv(METADATA_FILE)


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading trained model...")

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    1
)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()


# ==========================================
# IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# TEST
# ==========================================

results = []

test_images = list(
    TEST_DIR.glob("*.jpg")
)

print("\nTest images:", len(test_images))

print("\nPredictions:")
print("------------------------------------------")


for image_path in test_images:

    image_name = image_path.name

    match = metadata[
        metadata["image_name"] == image_name
    ]

    if len(match) == 0:
        continue

    actual = float(
        match.iloc[0]["intensity_knots"]
    )

    image = Image.open(
        image_path
    ).convert("L")

    image = transform(image)

    image = image.unsqueeze(0).to(DEVICE)


    with torch.no_grad():

        prediction = model(image)

        prediction = prediction.item()


    results.append({
        "image": image_name,
        "actual": actual,
        "predicted": prediction,
        "error": abs(actual - prediction)
    })


# ==========================================
# RESULTS
# ==========================================

results_df = pd.DataFrame(results)

mae = results_df["error"].mean()

rmse = np.sqrt(
    ((results_df["actual"] -
      results_df["predicted"]) ** 2).mean()
)


print(
    results_df.to_string(
        index=False,
        formatters={
            "actual": "{:.1f}".format,
            "predicted": "{:.1f}".format,
            "error": "{:.1f}".format
        }
    )
)


print("\n==========================================")
print("MODEL TEST RESULTS")
print("==========================================")

print(
    f"Mean Absolute Error (MAE): {mae:.2f} knots"
)

print(
    f"RMSE: {rmse:.2f} knots"
)

print(
    f"Test images evaluated: {len(results_df)}"
)

print("==========================================")


# ==========================================
# SAVE RESULTS
# ==========================================

output_file = (
    PROJECT
    / "data"
    / "processed"
    / "test_predictions.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print(
    "\nResults saved to:",
    output_file
)