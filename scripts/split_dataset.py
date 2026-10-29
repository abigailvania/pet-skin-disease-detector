
import os
import csv
import json
import random
from pathlib import Path
from collections import Counter
from PIL import Image

# ===================================
# CONFIGURATION
# ===================================

RAW_DIR = Path("dataset/raw")
IMAGE_DIR = RAW_DIR / "images"
DETECTION_LABELS_DIR = RAW_DIR / "detection_labels"
SEGMENTATION_MASKS_DIR = RAW_DIR / "segmentation_masks"

OUTPUT_DIR = Path("dataset/processed")
SPLITS_DIR = OUTPUT_DIR / "splits"

CLASSES = [
    "DEMODICOSIS",
    "DERMATITIS",
    "DERMATOFITOSIS",
    "HEALTHY",
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Label YOLO untuk HEALTHY boleh tidak tersedia.
# Segmentation mask boleh belum tersedia.
# Path dalam CSV relatif terhadap project root.

# ===================================
# VALIDATE CONFIGURATION
# ===================================

if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-6:
    raise ValueError("Split ratios must add up to 1.0.")

if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(
        f"Image directory not found: {IMAGE_DIR}"
    )

if not DETECTION_LABELS_DIR.is_dir():
    raise FileNotFoundError(
        f"Detection label directory not found: "
        f"{DETECTION_LABELS_DIR}"
    )

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"{OUTPUT_DIR} already exists. "
        "Back up or rename it before running this script."
    )

# ===================================
# FIND DETECTION LABEL
# ===================================

def find_detection_label(image_path, root):
    """
    Mencari label YOLO berdasarkan kelas dan nama gambar.
    Tidak melakukan pemindaian rekursif.
    """

    image_path = Path(image_path)
    root = Path(root)

    class_name = image_path.parent.name
    label_path = root / class_name / f"{image_path.stem}.txt"

    if label_path.is_file():
        return label_path.as_posix()

    # HEALTHY boleh tidak memiliki label YOLO.
    if class_name.upper() == "HEALTHY":
        return ""

    return ""


# ===================================
# FIND SEGMENTATION MASK
# ===================================

def find_segmentation_mask(image_path, root):
    """
    Mencari segmentation mask berdasarkan kelas dan nama gambar.
    Memeriksa ekstensi yang didukung tanpa memindai seluruh folder.

    Asumsi struktur:
    segmentation_masks/<CLASS>/<image_stem>.<mask_extension>
    """

    image_path = Path(image_path)
    root = Path(root)

    if not root.is_dir():
        return ""

    class_name = image_path.parent.name

    # Periksa ekstensi yang umum digunakan untuk mask.
    for extension in sorted(MASK_EXTENSIONS):
        mask_path = root / class_name / (
            f"{image_path.stem}{extension}"
        )

        if mask_path.is_file():
            return mask_path.as_posix()

    return ""


# ===================================
# SCAN IMAGES
# ===================================

records = []
class_counts = Counter()
seen_paths = set()
missing_detection_by_class = Counter()
missing_masks_by_class = Counter()

for class_name in CLASSES:
    class_dir = IMAGE_DIR / class_name

    if not class_dir.is_dir():
        raise FileNotFoundError(
            f"Class directory not found: {class_dir}"
        )

    class_images = sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image_path in class_images:
        relative_path = image_path.relative_to(IMAGE_DIR)
        path_key = relative_path.as_posix().casefold()

        if path_key in seen_paths:
            raise ValueError(
                f"Duplicate relative path: {relative_path}"
            )

        seen_paths.add(path_key)

        # Validate that the image can be opened.
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as error:
            raise ValueError(
                f"Unreadable image: {image_path}"
            ) from error

        # Find corresponding YOLO label.
        detection_label = find_detection_label(
            image_path,
            DETECTION_LABELS_DIR
        )

        if not detection_label:
            missing_detection_by_class[class_name] += 1

            if class_name != "HEALTHY":
                print(
                    f"WARNING: YOLO label missing: "
                    f"{relative_path}"
                )

        # Find corresponding segmentation mask.
        segmentation_mask = find_segmentation_mask(
            image_path,
            SEGMENTATION_MASKS_DIR
        )

        if not segmentation_mask:
            missing_masks_by_class[class_name] += 1

        # Store project-root-relative paths.
        records.append({
            "image": image_path.as_posix(),
            "relative_path": relative_path.as_posix(),
            "label": class_name,
            "detection_label": detection_label,
            "segmentation_mask": segmentation_mask,
        })

        class_counts[class_name] += 1

if not records:
    raise ValueError("No images found.")

# ===================================
# REPORT ORIGINAL COUNTS
# ===================================

print("\nOriginal dataset")
print("=" * 45)

for class_name in CLASSES:
    print(f"{class_name}: {class_counts[class_name]}")

print(f"TOTAL: {len(records)}")

print("\nMissing detection labels")
print("=" * 45)

for class_name in CLASSES:
    print(
        f"{class_name}: "
        f"{missing_detection_by_class[class_name]}"
    )

print("\nMissing segmentation masks")
print("=" * 45)

for class_name in CLASSES:
    print(
        f"{class_name}: "
        f"{missing_masks_by_class[class_name]}"
    )

# ===================================
# STRATIFIED SPLIT
# ===================================

rng = random.Random(SEED)

splits = {
    "train": [],
    "val": [],
    "test": [],
}

for class_name in CLASSES:
    class_records = [
        record
        for record in records
        if record["label"] == class_name
    ]

    rng.shuffle(class_records)

    total = len(class_records)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits["train"].extend(class_records[:train_end])
    splits["val"].extend(class_records[train_end:val_end])
    splits["test"].extend(class_records[val_end:])

for subset in splits.values():
    rng.shuffle(subset)

# ===================================
# CREATE MANIFESTS
# ===================================

SPLITS_DIR.mkdir(parents=True, exist_ok=True)

summary = {
    "seed": SEED,
    "ratios": {
        "train": TRAIN_RATIO,
        "val": VAL_RATIO,
        "test": TEST_RATIO,
    },
    "total_images": len(records),
    "splits": {},
}

fieldnames = [
    "image",
    "label",
    "detection_label",
    "segmentation_mask",
]

for split_name, subset in splits.items():
    csv_path = SPLITS_DIR / f"{split_name}.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in subset:
            writer.writerow({
                "image": record["image"],
                "label": record["label"],
                "detection_label": record["detection_label"],
                "segmentation_mask": record["segmentation_mask"],
            })

    split_counts = Counter(
        record["label"] for record in subset
    )

    summary["splits"][split_name] = {
        "total_images": len(subset),
        "class_counts": {
            class_name: split_counts[class_name]
            for class_name in CLASSES
        },
        "missing_detection_labels": sum(
            not record["detection_label"]
            for record in subset
        ),
        "missing_segmentation_masks": sum(
            not record["segmentation_mask"]
            for record in subset
        ),
    }

    print(f"\n{split_name.upper()}: {len(subset)} images")

    for class_name in CLASSES:
        print(f"  {class_name}: {split_counts[class_name]}")

# ===================================
# SAVE METADATA
# ===================================

with open(
    OUTPUT_DIR / "class_names.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(CLASSES, file, indent=4)

with open(
    OUTPUT_DIR / "split_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(summary, file, indent=4)

# ===================================
# FINAL REPORT
# ===================================

print("\n" + "=" * 45)
print("SPLIT MANIFESTS CREATED")
print("=" * 45)
print(f"Train CSV: {SPLITS_DIR / 'train.csv'}")
print(f"Val CSV:   {SPLITS_DIR / 'val.csv'}")
print(f"Test CSV:  {SPLITS_DIR / 'test.csv'}")
print(f"Summary:   {OUTPUT_DIR / 'split_summary.json'}")
print("No images, labels, or masks were copied or modified.")
print("Review missing-annotation counts before training.")
