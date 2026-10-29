
import os
import csv
import cv2
import numpy as np
from PIL import Image, ImageOps

# ===================================
# CONFIGURATION
# ===================================

DATASET_DIR = os.path.join("dataset", "raw2", "images")
OUTPUT_DIR = os.path.join("results", "dataset")

DETAIL_REPORT = os.path.join(
    OUTPUT_DIR, "image_quality_report.csv"
)

SUMMARY_REPORT = os.path.join(
    OUTPUT_DIR, "image_quality_summary.csv"
)

IMAGE_EXTENSION = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Thresholds
BLUR_THRESHOLD = 100
SHARP_THRESHOLD = 300

DARK_THRESHOLD = 50
BRIGHT_THRESHOLD = 205
LOW_CONTRAST_THRESHOLD = 25

# Resolusi
HIGH_MIN_LONG = 1920
HIGH_MIN_SHORT = 1080

MEDIUM_MIN_LONG = 1280
MEDIUM_MIN_SHORT = 720

# Ukuran seragam untuk analisis blur agar
# hasil lebih mudah dibandingkan antarresolusi.
ANALYSIS_SIZE = (512, 512)

# ===================================
# CHECK DATASET
# ===================================

if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset folder tidak ditemukan: "
        f"{os.path.abspath(DATASET_DIR)}"
    )

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================================
# FIND ALL IMAGES
# ===================================

images = sorted(
    os.path.join(root, file)
    for root, dirs, files in os.walk(DATASET_DIR)
    for file in files
    if os.path.splitext(file)[1].lower() in IMAGE_EXTENSION
)

print(f"Total images found: {len(images)}")

# ===================================
# ANALYZE IMAGE QUALITY
# ===================================

results = []
failed_images = []

for index, image_path in enumerate(images, start=1):

    relative_path = os.path.relpath(
        image_path, DATASET_DIR
    )

    relative_parts = relative_path.split(os.sep)

    category = (
        relative_parts[0]
        if len(relative_parts) > 1
        else "UNKNOWN"
    )

    try:
        # -------------------------------
        # READ IMAGE AND APPLY EXIF ROTATION
        # -------------------------------

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")

            width, height = image.size
            image_array = np.array(image)

        # -------------------------------
        # FILE SIZE
        # -------------------------------

        file_size_bytes = os.path.getsize(image_path)
        file_size_mb = file_size_bytes / (1024 * 1024)

        # -------------------------------
        # RESOLUTION CATEGORY
        # -------------------------------

        long_side = max(width, height)
        short_side = min(width, height)

        total_pixels = width * height

        if (
            long_side >= HIGH_MIN_LONG
            and short_side >= HIGH_MIN_SHORT
        ):
            resolution_category = "High"

        elif (
            long_side >= MEDIUM_MIN_LONG
            and short_side >= MEDIUM_MIN_SHORT
        ):
            resolution_category = "Medium"

        else:
            resolution_category = "Low"

        # -------------------------------
        # GRAYSCALE AND RESIZE
        # -------------------------------

        gray = cv2.cvtColor(
            image_array, cv2.COLOR_RGB2GRAY
        )

        gray_resized = cv2.resize(
            gray,
            ANALYSIS_SIZE,
            interpolation=cv2.INTER_AREA
        )

        # -------------------------------
        # BLUR DETECTION
        # -------------------------------

        laplacian_variance = cv2.Laplacian(
            gray_resized, cv2.CV_64F
        ).var()

        if laplacian_variance < BLUR_THRESHOLD:
            blur_status = "Blurry"

        elif laplacian_variance < SHARP_THRESHOLD:
            blur_status = "Moderate"

        else:
            blur_status = "Sharp"

        # -------------------------------
        # BRIGHTNESS
        # -------------------------------

        brightness = float(np.mean(gray_resized))

        if brightness < DARK_THRESHOLD:
            brightness_status = "Too dark"

        elif brightness > BRIGHT_THRESHOLD:
            brightness_status = "Too bright"

        else:
            brightness_status = "Normal"

        # -------------------------------
        # CONTRAST
        # -------------------------------

        contrast = float(np.std(gray_resized))

        if contrast < LOW_CONTRAST_THRESHOLD:
            contrast_status = "Low contrast"

        else:
            contrast_status = "Normal"

        # -------------------------------
        # REVIEW FLAGS
        # -------------------------------

        issues = []

        if blur_status == "Blurry":
            issues.append("Blurry")

        if resolution_category == "Low":
            issues.append("Low resolution")

        if brightness_status != "Normal":
            issues.append(brightness_status)

        if contrast_status == "Low contrast":
            issues.append("Low contrast")

        review_status = (
            "Review"
            if issues
            else "No automatic flag"
        )

        # -------------------------------
        # SAVE RESULT
        # -------------------------------

        results.append({
            "file": os.path.basename(image_path),
            "path": relative_path,
            "class": category,
            "width": width,
            "height": height,
            "total_pixels": total_pixels,
            "resolution_category": resolution_category,
            "file_size_mb": round(file_size_mb, 4),
            "laplacian_variance": round(
                float(laplacian_variance), 2
            ),
            "blur_status": blur_status,
            "brightness": round(brightness, 2),
            "brightness_status": brightness_status,
            "contrast_std": round(contrast, 2),
            "contrast_status": contrast_status,
            "issues": "; ".join(issues),
            "review_status": review_status,
        })

    except Exception as error:
        failed_images.append({
            "file": os.path.basename(image_path),
            "path": relative_path,
            "class": category,
            "error": str(error),
        })

        print(f"Failed: {relative_path} | {error}")

    if index % 500 == 0 or index == len(images):
        print(f"Processed: {index}/{len(images)}")

# ===================================
# SAVE DETAIL REPORT
# ===================================

detail_fields = [
    "file",
    "path",
    "class",
    "width",
    "height",
    "total_pixels",
    "resolution_category",
    "file_size_mb",
    "laplacian_variance",
    "blur_status",
    "brightness",
    "brightness_status",
    "contrast_std",
    "contrast_status",
    "issues",
    "review_status",
]

with open(
    DETAIL_REPORT,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:
    writer = csv.DictWriter(
        file, fieldnames=detail_fields
    )
    writer.writeheader()
    writer.writerows(results)

# ===================================
# GENERATE SUMMARY
# ===================================

summary_fields = [
    "class",
    "total_images",
    "high_resolution",
    "high_resolution_pct",
    "medium_resolution",
    "medium_resolution_pct",
    "low_resolution",
    "low_resolution_pct",
    "blurry",
    "blurry_pct",
    "moderate_sharpness",
    "moderate_sharpness_pct",
    "sharp",
    "sharp_pct",
    "too_dark",
    "too_dark_pct",
    "too_bright",
    "too_bright_pct",
    "low_contrast",
    "low_contrast_pct",
    "flagged_for_review",
    "flagged_for_review_pct",
]

summary_rows = []

categories = sorted(set(
    row["class"] for row in results
))

# Overall summary + per-class summary
for category in ["ALL"] + categories:

    if category == "ALL":
        group = results
    else:
        group = [
            row for row in results
            if row["class"] == category
        ]

    total = len(group)

    if total == 0:
        continue

    def count_status(field, value):
        return sum(
            row[field] == value
            for row in group
        )

    def count_pct(count):
        return round(count / total * 100, 2)

    high = count_status("resolution_category", "High")
    medium = count_status("resolution_category", "Medium")
    low = count_status("resolution_category", "Low")

    blurry = count_status("blur_status", "Blurry")
    moderate = count_status("blur_status", "Moderate")
    sharp = count_status("blur_status", "Sharp")

    dark = count_status("brightness_status", "Too dark")
    bright = count_status("brightness_status", "Too bright")

    low_contrast = count_status(
        "contrast_status", "Low contrast"
    )

    flagged = count_status("review_status", "Review")

    summary_rows.append({
        "class": category,
        "total_images": total,

        "high_resolution": high,
        "high_resolution_pct": count_pct(high),

        "medium_resolution": medium,
        "medium_resolution_pct": count_pct(medium),

        "low_resolution": low,
        "low_resolution_pct": count_pct(low),

        "blurry": blurry,
        "blurry_pct": count_pct(blurry),

        "moderate_sharpness": moderate,
        "moderate_sharpness_pct": count_pct(moderate),

        "sharp": sharp,
        "sharp_pct": count_pct(sharp),

        "too_dark": dark,
        "too_dark_pct": count_pct(dark),

        "too_bright": bright,
        "too_bright_pct": count_pct(bright),

        "low_contrast": low_contrast,
        "low_contrast_pct": count_pct(low_contrast),

        "flagged_for_review": flagged,
        "flagged_for_review_pct": count_pct(flagged),
    })

with open(
    SUMMARY_REPORT,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:
    writer = csv.DictWriter(
        file, fieldnames=summary_fields
    )
    writer.writeheader()
    writer.writerows(summary_rows)

# ===================================
# SAVE FAILED IMAGES
# ===================================

if failed_images:
    failed_report = os.path.join(
        OUTPUT_DIR, "failed_image_quality.csv"
    )

    with open(
        failed_report,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "file", "path", "class", "error"
            ]
        )
        writer.writeheader()
        writer.writerows(failed_images)

    print(f"Failed-image report: {failed_report}")

# ===================================
# DISPLAY SUMMARY
# ===================================

print("\n" + "=" * 60)
print("IMAGE QUALITY ANALYSIS SUMMARY")
print("=" * 60)

for row in summary_rows:
    print(f"\nClass: {row['class']}")
    print(f"  Total images     : {row['total_images']}")
    print(
        f"  High resolution  : {row['high_resolution']} "
        f"({row['high_resolution_pct']}%)"
    )
    print(
        f"  Medium resolution: {row['medium_resolution']} "
        f"({row['medium_resolution_pct']}%)"
    )
    print(
        f"  Low resolution   : {row['low_resolution']} "
        f"({row['low_resolution_pct']}%)"
    )
    print(
        f"  Blurry           : {row['blurry']} "
        f"({row['blurry_pct']}%)"
    )
    print(
        f"  Moderate sharpness: {row['moderate_sharpness']} "
        f"({row['moderate_sharpness_pct']}%)"
    )
    print(
        f"  Sharp            : {row['sharp']} "
        f"({row['sharp_pct']}%)"
    )
    print(
        f"  Too dark         : {row['too_dark']} "
        f"({row['too_dark_pct']}%)"
    )
    print(
        f"  Too bright       : {row['too_bright']} "
        f"({row['too_bright_pct']}%)"
    )
    print(
        f"  Low contrast     : {row['low_contrast']} "
        f"({row['low_contrast_pct']}%)"
    )
    print(
        f"  Flagged for review: {row['flagged_for_review']} "
        f"({row['flagged_for_review_pct']}%)"
    )

print("\n" + "=" * 60)
print(f"Detail report : {DETAIL_REPORT}")
print(f"Summary report: {SUMMARY_REPORT}")
print("Analysis completed. No images were modified or deleted.")
