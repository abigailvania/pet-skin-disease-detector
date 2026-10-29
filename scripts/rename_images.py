
import os
import csv

# ===================================
# CONFIGURATION
# ===================================

IMAGE_ROOT = "dataset/raw/images"
MAPPING_FILE = "results/dataset/rename_mapping.csv"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ===================================
# VALIDATE DIRECTORY
# ===================================

if not os.path.isdir(IMAGE_ROOT):
    raise FileNotFoundError(
        f"Folder tidak ditemukan: {os.path.abspath(IMAGE_ROOT)}"
    )

# ===================================
# COLLECT ALL IMAGES
# ===================================

image_files = []

for folder in sorted(os.listdir(IMAGE_ROOT)):
    folder_path = os.path.join(IMAGE_ROOT, folder)

    if not os.path.isdir(folder_path):
        continue

    for filename in sorted(os.listdir(folder_path)):
        old_path = os.path.join(folder_path, filename)

        if (
            os.path.isfile(old_path)
            and os.path.splitext(filename)[1].lower() in VALID_EXTENSIONS
        ):
            image_files.append({
                "folder": folder,
                "old_name": filename,
                "old_path": old_path,
            })

if not image_files:
    raise ValueError("Tidak ada gambar yang ditemukan.")

# ===================================
# CREATE GLOBAL RENAME MAPPING
# ===================================

mapping = []

for index, item in enumerate(image_files, start=1):
    extension = os.path.splitext(item["old_name"])[1].lower()
    new_name = f"pet_{index:04d}{extension}"

    mapping.append({
        "folder": item["folder"],
        "old_name": item["old_name"],
        "new_name": new_name,
        "old_path": item["old_path"],
        "new_path": os.path.join(
            os.path.dirname(item["old_path"]),
            new_name
        ),
    })

# ===================================
# CHECK FOR TARGET CONFLICTS
# ===================================

source_paths = {item["old_path"] for item in mapping}

for item in mapping:
    if (
        os.path.exists(item["new_path"])
        and item["new_path"] not in source_paths
    ):
        raise FileExistsError(
            f"Nama target sudah digunakan: {item['new_path']}"
        )

# ===================================
# SAVE MAPPING CSV
# ===================================

os.makedirs(os.path.dirname(MAPPING_FILE), exist_ok=True)

with open(MAPPING_FILE, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["folder", "old_name", "new_name"]
    )
    writer.writeheader()

    for item in mapping:
        writer.writerow({
            "folder": item["folder"],
            "old_name": item["old_name"],
            "new_name": item["new_name"],
        })

# ===================================
# RENAME VIA TEMPORARY NAMES
# ===================================

temp_files = []

try:
    # Stage 1: Move every image to a unique temporary name.
    for index, item in enumerate(mapping, start=1):
        temp_name = f"__temp_rename_{index:08d}__"
        temp_path = os.path.join(
            os.path.dirname(item["old_path"]),
            temp_name
        )

        os.rename(item["old_path"], temp_path)
        temp_files.append((temp_path, item["new_path"]))

    # Stage 2: Apply the final names.
    for temp_path, new_path in temp_files:
        os.rename(temp_path, new_path)

except Exception:
    print(
        "WARNING: Rename tidak selesai. "
        "Periksa file yang mungkin masih menggunakan nama sementara "
        "sebelum menjalankan ulang script."
    )
    raise

# ===================================
# SUMMARY
# ===================================

print("\nRename berhasil!")

for folder in sorted({item["folder"] for item in mapping}):
    count = sum(1 for item in mapping if item["folder"] == folder)
    print(f"{folder}: {count} gambar")

print(f"\nTotal gambar: {len(mapping)}")
print(f"Mapping CSV: {os.path.abspath(MAPPING_FILE)}")
