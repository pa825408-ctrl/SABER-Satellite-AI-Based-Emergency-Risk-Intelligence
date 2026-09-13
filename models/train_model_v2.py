import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path


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

DATASET_DIR = PROJECT / "data" / "dataset"

MODEL_OUTPUT = (
    PROJECT
    / "models"
    / "best_model_v2.pth"
)

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.00001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("======================================")
print("CYCLONE INTENSITY AI - VERSION 2")
print("======================================")
print("Device:", DEVICE)


# ==========================================
# LOAD METADATA
# ==========================================

metadata = pd.read_csv(METADATA_FILE)

print("\nTotal images:", len(metadata))

print(
    "Intensity range:",
    metadata["intensity_knots"].min(),
    "to",
    metadata["intensity_knots"].max(),
    "knots"
)


# ==========================================
# NORMALIZATION VALUES
# ==========================================

MIN_INTENSITY = metadata["intensity_knots"].min()
MAX_INTENSITY = metadata["intensity_knots"].max()

RANGE = MAX_INTENSITY - MIN_INTENSITY


# ==========================================
# DATASET
# ==========================================

class CycloneDataset(Dataset):

    def __init__(self, dataframe, split):

        self.data = dataframe.reset_index(drop=True)

        self.split_dir = DATASET_DIR / split

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),

            transforms.RandomHorizontalFlip(p=0.5),

            transforms.RandomRotation(10),

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

        image_path = (
            self.split_dir
            / row["image_name"]
        )

        image = Image.open(
            image_path
        ).convert("L")

        image = self.transform(image)

        intensity = float(
            row["intensity_knots"]
        )

        # Normalize intensity to 0-1
        label = (
            intensity - MIN_INTENSITY
        ) / RANGE

        label = torch.tensor(
            label,
            dtype=torch.float32
        )

        return image, label


# ==========================================
# LOAD SPLITS
# ==========================================

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
                float(
                    match.iloc[0][
                        "intensity_knots"
                    ]
                )
            })

    return pd.DataFrame(rows)


train_df = load_split("train")

val_df = load_split("validation")

print("\nTraining images:", len(train_df))

print(
    "Validation images:",
    len(val_df)
)


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
# MODEL
# ==========================================

print("\nLoading ResNet18...")

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)


# Fine-tune the last layers
for param in model.parameters():

    param.requires_grad = False


for param in model.layer4.parameters():

    param.requires_grad = True


model.fc = nn.Linear(
    model.fc.in_features,
    1
)


model = model.to(DEVICE)


# ==========================================
# LOSS
# ==========================================

criterion = nn.MSELoss()


# ==========================================
# OPTIMIZER
# ==========================================

trainable_parameters = filter(
    lambda p: p.requires_grad,
    model.parameters()
)

optimizer = torch.optim.Adam(
    trainable_parameters,
    lr=LEARNING_RATE
)


# ==========================================
# TRAINING
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


    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}"
    )


    # ------------------------------
    # SAVE BEST
    # ------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "min_intensity":
                    float(MIN_INTENSITY),

                "max_intensity":
                    float(MAX_INTENSITY)
            },
            MODEL_OUTPUT
        )

        print(
            "  -> Best model saved!"
        )


# ==========================================
# COMPLETE
# ==========================================

print("\n======================================")
print("VERSION 2 TRAINING COMPLETE")
print("======================================")

print(
    "Best model:",
    MODEL_OUTPUT
)

print(
    "Intensity normalization:",
    MIN_INTENSITY,
    "-",
    MAX_INTENSITY
)