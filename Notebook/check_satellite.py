from pathlib import Path
from PIL import Image

SATELLITE_DIR = Path(
    "data/satellite/insat3d/insat3d_ir_cyclone_ds"
)

print("Satellite directory:")
print(SATELLITE_DIR)

files = []

for extension in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]:
    files.extend(SATELLITE_DIR.rglob(extension))

print("\nNumber of images:", len(files))

print("\nFirst 10 images:")

for file in files[:10]:
    print(file)