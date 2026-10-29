
from pathlib import Path
from PIL import Image, ImageOps
import csv

# ===================================
# CONFIGURATION
# ===================================

INPUT_DIR = Path("dataset/raw/images")
OUTPUT_DIR = Path("dataset/images/processed")

MAPPING_FILE = Path("results/dataset/rename_mapping.csv")

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"
}

# ===================================
# VALIDATE INPUT
# ===================================

if not INPUT_DIR.is_dir():
    raise FileNotFoundError(
        f"Dataset folder not found: {INPUT_DIR.resolve()}"
    )

disease_folders = sorted(
    path for path in INPUT_DIR.iterdir()
    if path.is_dir()
)

if not disease_folders:
    raise ValueError("No disease folders found.")

# Collect images in deterministic order.
images = []

for folder in disease_folders:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            images.append((folder.name, path))

if not images:
    raise ValueError("No supported images found.")

# Do not accidentally overwrite an existing output dataset.
if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(
        f"Output folder is not empty: {OUTPUT_DIR.resolve()}\n"
        "Choose a new OUTPUT_DIR or move the existing output first."
    )

# ===================================
# CONVERT AND RENAME
# ===================================

mapping = []

for index, (disease, source_path) in enumerate(images, start=1):
    output_folder = OUTPUT_DIR / disease
    output_folder.mkdir(parents=True, exist_ok=True)

    new_name = f"pet_{index:05d}.jpg"
    output_path = output_folder / new_name

    try:
        with Image.open(source_path) as image:
            image = ImageOps.exif_transpose(image)

            # JPG does not support transparency.
            # Composite transparent areas onto a white background.
            if image.mode in ("RGBA", "LA") or (
                image.mode == "P" and "transparency" in image.info
            ):
                rgba = image.convert("RGBA")
                background = Image.new("RGB", rgba.size, "white")
                background.paste(rgba, mask=rgba.getchannel("A"))
                image = background
            else:
                image = image.convert("RGB")

            image.save(
                output_path,
                format="JPEG",
                quality=95,
                optimize=True
            )

        mapping.append({
            "folder": disease,
            "original_path": str(source_path),
            "new_name": new_name,
            "output_path": str(output_path),
            "status": "success"
        })

    except Exception as error:
        mapping.append({
            "folder": disease,
            "original_path": str(source_path),
            "new_name": new_name,
            "output_path": str(output_path),
            "status": f"failed: {error}"
        })

    if index % 100 == 0 or index == len(images):
        print(f"Processed: {index}/{len(images)}")

# ===================================
# SAVE MAPPING CSV
# ===================================

MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)

with MAPPING_FILE.open(
    "w", newline="", encoding="utf-8-sig"
) as file:
    fields = [
        "folder", "original_path", "new_name",
        "output_path", "status"
    ]
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(mapping)

# ===================================
# SUMMARY
# ===================================

success_count = sum(
    row["status"] == "success" for row in mapping
)
failed_count = len(mapping) - success_count

print("\n=== SUMMARY ===")
print(f"Total images: {len(images)}")
print(f"Converted successfully: {success_count}")
print(f"Failed: {failed_count}")
print(f"Output folder: {OUTPUT_DIR.resolve()}")
print(f"Mapping CSV: {MAPPING_FILE.resolve()}")
