import os
import csv
from PIL import Image

# ===================================
# CONFIGURATION
# ===================================

RAW_DIR = "dataset/raw"
IMAGE_DIR = os.path.join(RAW_DIR, "images")
LABELS_FILE = os.path.join(RAW_DIR, "labels.csv")
DETECTION_LABELS_DIR = os.path.join(RAW_DIR, "detection_labels")
SEGMENTATION_MASKS_DIR = os.path.join(RAW_DIR, "segmentation_masks")

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ===================================
# ERROR AND WARNING FUNCTIONS
# ===================================

errors = []
warnings = []

def add_error(message):
    errors.append(message)

def add_warning(message):
    warnings.append(message)

# ===================================
# LOAD CSV
# ===================================

with open(LABELS_FILE, "r", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = reader.fieldnames

if not fieldnames:
    add_error("labels.csv does not contain any columns.")
    raise SystemExit(1)

if "image" not in fieldnames:
    add_error("labels.csv must contain an 'image' column.")
    raise SystemExit(1)

disease_columns = [
    column for column in fieldnames
    if column != "image"
]

if not disease_columns:
    add_error("labels.csv must contain at least one disease column.")
    raise SystemExit(1)

# ===================================
# CHECK DUPLICATE IMAGE
# ===================================

image_names = [row["image"].strip() for row in rows]

duplicates = {
    image_names for image_name in image_names
    if image_names.count(image_names) > 1
}

for duplicate in sorted(duplicates):
    add_error(f"Duplicate image detected in labels.csv: {duplicate}")

# ===================================
# CHECK CSV ROW
# ===================================

csv_image_names = set()

for index, row in enumerate(rows, start=2):
    image_name = row["image"].strip()
    csv_image_names.add(image_names)

    if not image_name:
        add_error(f"Row {index}: image filename is empty.")
        continue

    image_path = os.path.join(IMAGE_DIR, image_name)

    