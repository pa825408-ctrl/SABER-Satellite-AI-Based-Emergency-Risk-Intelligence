import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path


# ==========================================
# 1. SETTINGS
# ==========================================

PROJECT = Path(".")

METADATA_FILE = (
    PROJECT
    / "data"
    / "processed"
    / "insat_ir_metadata.csv"
)

DATASET_DIR = PROJECT / "data" / "dataset"

MODEL_OUTPUT = PROJECT / "models" / "best_model.pth"

BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 0.0001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("======================================")
print("CYCLONE INTENSITY AI TRAINING")
print("======================================")
print("Device:", DEVICE)


# ==========================================
# 2. LOAD METADATA
# ==========================================

metadata = pd.read_csv(METADATA_FILE)

print("\nTotal images:", len(metadata))
print("Intensity range:",
      metadata["intensity_knots"].min(),
      "to",
      metadata["intensity_knots"].max(),
      "knots")


# ==========================================
# 3. DATASET CLASS
# ==========================================

class CycloneDataset(Dataset):

    def __init__(self, dataframe, split):

        self.data = dataframe.reset_index(drop=True)

        self.split_dir = DATASET_DIR / split

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = self.split_dir / row["image_name"]

        image = Image.open(image_path).convert("L")

        image = self.transform(image)

        label = torch.tensor(
            row["intensity_knots"],
            dtype=torch.float32
        )

        return image, label


# ==========================================
# 4. CREATE TRAIN / VALIDATION DATA
# ==========================================

import os

def load_split(split):

    split_dir = DATASET_DIR / split

    rows = []

    for image_file in split_dir.glob("*.jpg"):

        image_name = image_file.name

        match = metadata[
            metadata["image_name"] == image_name
        ]

        if len(match) > 0:

            rows.append({
                "image_name": image_name,
                "intensity_knots":
                    float(match.iloc[0]["intensity_knots"])
            })

    return pd.DataFrame(rows)


train_df = load_split("train")
val_df = load_split("validation")

print("\nTraining images:", len(train_df))
print("Validation images:", len(val_df))


train_dataset = CycloneDataset(
    train_df,
    "train"
)

val_dataset = CycloneDataset(
    val_df,
    "validation"
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 5. LOAD RESNET18
# ==========================================

print("\nLoading ResNet18...")

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Freeze most layers
for param in model.parameters():
    param.requires_grad = False


# Replace final layer
model.fc = nn.Linear(
    model.fc.in_features,
    1
)

model = model.to(DEVICE)


# ==========================================
# 6. LOSS + OPTIMIZER
# ==========================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 7. TRAINING
# ==========================================

best_val_loss = float("inf")


for epoch in range(EPOCHS):

    # ------------------------------
    # TRAIN
    # ------------------------------

    model.train()

    train_loss = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        outputs = outputs.squeeze(1)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # ------------------------------
    # VALIDATION
    # ------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            outputs = outputs.squeeze(1)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)


    # Calculate MAE

    val_mae = val_loss ** 0.5


    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.2f} "
        f"Val Loss: {val_loss:.2f}"
    )


    # ------------------------------
    # SAVE BEST MODEL
    # ------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_OUTPUT
        )

        print("  -> Best model saved!")


# ==========================================
# 8. FINISHED
# ==========================================

print("\n======================================")
print("TRAINING COMPLETE")
print("======================================")

print("Best model:")
print(MODEL_OUTPUT)