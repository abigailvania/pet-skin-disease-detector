from pathlib import Path
import hashlib
import csv
import imagehash
from PIL import Image
from itertools import combinations

# ===================================
# CONFIGURATION
# ===================================

DATASET_DIR = Path("dataset/raw/images/DEMODICOSIS")
IMAGE_EXTENSION = {".jpg", ".jpeg", ".png", ".webp"}

OUTPUT_DIR = Path("results/dataset")
REPORT_PATH = OUTPUT_DIR / "duplicate_report.csv"

# Semakin kecil threshold, semakin mirip gambar yang harus cocok.
# 0  = hash identik
# 4  = sangat mirip
# 8  = kandidat gambar mirip
# 10+ = lebih longgar, berpotensi lebih banyak false positive
PHASH_THRESHOLD = 8

# ===================================
# CALCULATE MD5 HASH
# ===================================

def calculate_hash(file_path):
    md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        while True:
            data = f.read(8192)
            if not data:
                break
            md5.update(data)

    return md5.hexdigest()


# ===================================
# CALCULATE PERCEPTUAL HASH (pHash)
# ===================================

def calculate_phash(file_path):
    with Image.open(file_path) as image:
        # Mengubah orientasi berdasarkan metadata EXIF
        image = image.convert("RGB")
        return imagehash.phash(image)


# ===================================
# FIND ALL IMAGES
# ===================================

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder tidak ditemukan: {DATASET_DIR.resolve()}"
    )

images = sorted(
    path for path in DATASET_DIR.rglob("*")
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSION
)

print(f"Total images: {len(images)}")

# ===================================
# CALCULATE BOTH HASHES
# ===================================

image_data = []
failed_images = []

for index, image_path in enumerate(images, start=1):
    try:
        md5_hash = calculate_hash(image_path)
        phash_value = calculate_phash(image_path)

        image_data.append({
            "path": image_path,
            "md5": md5_hash,
            "phash": phash_value,
        })

    except Exception as error:
        failed_images.append({
            "file": str(image_path),
            "error": str(error),
        })
        print(f"Failed to read: {image_path} | {error}")

    if index % 500 == 0 or index == len(images):
        print(f"Processed: {index}/{len(images)}")

# ===================================
# FIND EXACT AND VISUAL DUPLICATES
# ===================================

duplicate_pairs = []

# Membandingkan setiap pasangan gambar.
# Kompleksitas O(n^2), jadi prosesnya bisa lama
# jika jumlah gambar sangat besar.

for image1, image2 in combinations(image_data, 2):
    md5_same = image1["md5"] == image2["md5"]

    # Jarak Hamming antara dua perceptual hash.
    # Semakin kecil jaraknya, semakin mirip gambarnya.
    phash_distance = image1["phash"] - image2["phash"]

    if md5_same or phash_distance <= PHASH_THRESHOLD:
        path1 = image1["path"]
        path2 = image2["path"]

        relative1 = path1.relative_to(DATASET_DIR)
        relative2 = path2.relative_to(DATASET_DIR)

        category1 = (
            relative1.parts[0] if relative1.parts else ""
        )
        category2 = (
            relative2.parts[0] if relative2.parts else ""
        )

        if md5_same:
            duplicate_type = "Exact duplicate (MD5)"
        else:
            duplicate_type = "Visual similarity (pHash)"

        duplicate_pairs.append({
            "duplicate_type": duplicate_type,
            "md5_same": md5_same,
            "phash_distance": phash_distance,
            "file1": str(path1),
            "file2": str(path2),
            "category1": category1,
            "category2": category2,
            "md5_1": image1["md5"],
            "md5_2": image2["md5"],
            "phash_1": str(image1["phash"]),
            "phash_2": str(image2["phash"]),
        })

# ===================================
# DISPLAY RESULTS
# ===================================

print()
print(f"Duplicate pairs found: {len(duplicate_pairs)}")

for index, pair in enumerate(duplicate_pairs, start=1):
    print(f"\nPair {index}: {pair['duplicate_type']}")
    print(f"  File 1: {pair['file1']}")
    print(f"  File 2: {pair['file2']}")
    print(f"  MD5 identical: {pair['md5_same']}")
    print(f"  pHash distance: {pair['phash_distance']}")

# ===================================
# SAVE CSV REPORT
# ===================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "duplicate_type",
    "md5_same",
    "phash_distance",
    "file1",
    "file2",
    "category1",
    "category2",
    "md5_1",
    "md5_2",
    "phash_1",
    "phash_2",
]

with open(REPORT_PATH, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(duplicate_pairs)

# Simpan juga daftar gambar yang gagal dibaca
if failed_images:
    failed_report = OUTPUT_DIR / "failed_images.csv"

    with open(
        failed_report, "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=["file", "error"]
        )
        writer.writeheader()
        writer.writerows(failed_images)

    print(f"Failed-image report: {failed_report}")

print(f"Report stored at: {REPORT_PATH.resolve()}")