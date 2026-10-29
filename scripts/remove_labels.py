import os

# ===================================
# CONFIGURATION
# ===================================

IMAGE_DIR = os.path.join("dataset", "raw", "images")
LABEL_DIR = os.path.join("dataset", "raw", "detection_labels")

IMAGE_EXTENSION = {".jpg", ".jpeg", ".png", ".webp"}
LABEL_EXTENSION = ".txt"

# ===================================
# CHECK DIRECTORIES
# ===================================

if not os.path.isdir(IMAGE_DIR):
    raise FileNotFoundError(
        f"Image folder tidak ditemukan: {os.path.abspath(IMAGE_DIR)}"
    )

if not os.path.isdir(LABEL_DIR):
    raise FileNotFoundError(
        f"Label folder tidak ditemukan: {os.path.abspath(LABEL_DIR)}"
    )

# ===================================
# FIND ALL IMAGES
# ===================================

image_names = set()

for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        name, extension = os.path.splitext(file)

        if extension.lower() in IMAGE_EXTENSION:
            relative_folder = os.path.relpath(root, IMAGE_DIR)

            image_key = (
                relative_folder,
                name.lower()
            )

            image_names.add(image_key)

print(f"Total images found: {len(image_names)}")

# ===================================
# FIND LABELS WITHOUT IMAGES
# ===================================

orphan_labels = []

for root, dirs, files in os.walk(LABEL_DIR):
    for file in files:
        name, extension = os.path.splitext(file)

        if extension.lower() != LABEL_EXTENSION:
            continue

        relative_folder = os.path.relpath(root, LABEL_DIR)

        label_key = (
            relative_folder,
            name.lower()
        )

        if label_key not in image_names:
            orphan_labels.append(os.path.join(root, file))

# ===================================
# DISPLAY LABELS TO BE REMOVED
# ===================================

print()
print(f"Labels without matching images: {len(orphan_labels)}")

for label in orphan_labels:
    print(f"  {label}")

# ===================================
# REMOVE ORPHAN LABELS
# ===================================

for label in orphan_labels:
    os.remove(label)

print()
print(f"Orphan labels removed: {len(orphan_labels)}")
print("Label cleanup completed.")