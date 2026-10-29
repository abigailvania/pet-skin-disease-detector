
import os
import csv
import uuid
from collections import Counter

# ===================================
# CONFIGURATION
# ===================================

LABEL_ROOT = "dataset/raw/detection_labels"
MAPPING_FILE = "results/dataset/rename_mapping.csv"
LABEL_MAPPING_FILE = "results/dataset/rename_label_mapping.csv"

# ===================================
# VALIDATE
# ===================================

if not os.path.isdir(LABEL_ROOT):
    raise FileNotFoundError(
        f"Folder label tidak ditemukan: {os.path.abspath(LABEL_ROOT)}"
    )

if not os.path.isfile(MAPPING_FILE):
    raise FileNotFoundError(
        f"Mapping CSV tidak ditemukan: {os.path.abspath(MAPPING_FILE)}"
    )

# ===================================
# READ IMAGE MAPPING
# ===================================

with open(
    MAPPING_FILE, "r", newline="", encoding="utf-8-sig"
) as file:
    reader = csv.DictReader(file)

    required_columns = {"folder", "old_name", "new_name"}

    if not required_columns.issubset(reader.fieldnames or []):
        raise ValueError(
            "Kolom rename_mapping.csv harus mencakup: "
            "folder, old_name, new_name"
        )

    image_mapping = list(reader)

if not image_mapping:
    raise ValueError("Mapping CSV kosong.")

# ===================================
# BUILD LABEL MAPPING
# ===================================

label_mapping = []

for item in image_mapping:
    folder = item["folder"].strip()
    old_name = item["old_name"].strip()
    new_name = item["new_name"].strip()

    if not folder or not old_name or not new_name:
        raise ValueError(
            f"Ada baris mapping yang kosong: {item}"
        )

    # Pastikan nama file tidak mengandung path.
    if (
        os.path.basename(old_name) != old_name
        or os.path.basename(new_name) != new_name
    ):
        raise ValueError(
            f"Nama file tidak valid di mapping: {item}"
        )

    old_stem = os.path.splitext(old_name)[0]
    new_stem = os.path.splitext(new_name)[0]

    folder_path = os.path.join(LABEL_ROOT, folder)

    old_path = os.path.join(folder_path, old_stem + ".txt")
    new_path = os.path.join(folder_path, new_stem + ".txt")

    label_mapping.append({
        "folder": folder,
        "old_name": old_stem + ".txt",
        "new_name": new_stem + ".txt",
        "old_path": old_path,
        "new_path": new_path,
        "status": "PENDING",
    })

# ===================================
# CHECK DUPLICATE MAPPINGS
# ===================================

source_keys = [
    os.path.normcase(os.path.abspath(item["old_path"]))
    for item in label_mapping
]

target_keys = [
    os.path.normcase(os.path.abspath(item["new_path"]))
    for item in label_mapping
]

duplicate_sources = {
    key for key, count in Counter(source_keys).items()
    if count > 1
}

duplicate_targets = {
    key for key, count in Counter(target_keys).items()
    if count > 1
}

if duplicate_sources:
    raise ValueError(
        f"Ada {len(duplicate_sources)} sumber label duplikat. "
        "Rename dibatalkan."
    )

if duplicate_targets:
    raise ValueError(
        f"Ada {len(duplicate_targets)} target label duplikat. "
        "Rename dibatalkan."
    )

# ===================================
# INSPECT CURRENT FILES
# ===================================

pending = []
missing_labels = []
unchanged = []

for item in label_mapping:
    old_path = item["old_path"]
    new_path = item["new_path"]

    if os.path.abspath(old_path) == os.path.abspath(new_path):
        item["status"] = "UNCHANGED"
        unchanged.append(item)
        continue

    if not os.path.isdir(os.path.dirname(old_path)):
        item["status"] = "FOLDER_NOT_FOUND"
        missing_labels.append(item)
        continue

    if os.path.isfile(old_path):
        item["status"] = "PENDING"
        pending.append(item)

    elif os.path.isfile(new_path):
        # Kemungkinan sudah pernah di-rename.
        item["status"] = "SOURCE_MISSING_TARGET_EXISTS"

    elif item["folder"].upper() == "HEALTHY":
        # Healthy boleh tidak mempunyai anotasi.
        item["status"] = "HEALTHY_NO_LABEL"

    else:
        item["status"] = "SOURCE_NOT_FOUND"
        missing_labels.append(item)

# ===================================
# CHECK TARGET CONFLICTS
# ===================================

source_paths = {
    os.path.normcase(os.path.abspath(item["old_path"]))
    for item in pending
}

conflicts = []

for item in pending:
    target_key = os.path.normcase(
        os.path.abspath(item["new_path"])
    )

    if (
        os.path.exists(item["new_path"])
        and target_key not in source_paths
    ):
        item["status"] = "CONFLICT"
        conflicts.append(item)

# ===================================
# SAVE REPORT
# ===================================

def save_report():
    output_dir = os.path.dirname(LABEL_MAPPING_FILE)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(
        LABEL_MAPPING_FILE, "w",
        newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "folder",
                "old_name",
                "new_name",
                "status",
            ],
        )
        writer.writeheader()

        for item in label_mapping:
            writer.writerow({
                "folder": item["folder"],
                "old_name": item["old_name"],
                "new_name": item["new_name"],
                "status": item["status"],
            })

# ===================================
# PREVIEW
# ===================================

print("\n=== RENAME LABEL YOLO ===")
print(f"Total mapping: {len(label_mapping)}")
print(f"Siap di-rename: {len(pending) - len(conflicts)}")
print(f"Konflik: {len(conflicts)}")
print(f"Label tidak ditemukan: {len(missing_labels)}")

for item in label_mapping[:20]:
    print(
        f'{item["folder"]}: '
        f'{item["old_name"]} -> {item["new_name"]} '
        f'[{item["status"]}]'
    )

# ===================================
# STOP IF CONFLICTS EXIST
# ===================================

if conflicts:
    save_report()

    print("\nContoh konflik:")

    for item in conflicts[:20]:
        print(
            f'{item["folder"]}: '
            f'{item["old_name"]} -> {item["new_name"]}'
        )

    raise FileExistsError(
        f"Ada {len(conflicts)} konflik nama label. "
        f"Periksa {os.path.abspath(LABEL_MAPPING_FILE)}. "
        "Tidak ada label yang di-rename."
    )

# ===================================
# RENAME VIA TEMPORARY NAMES
# ===================================

temp_files = []

try:
    # Stage 1:
    # Pindahkan semua sumber ke nama sementara.
    for item in pending:
        temp_path = os.path.join(
            os.path.dirname(item["old_path"]),
            f"__rename_tmp_{uuid.uuid4().hex}.txt"
        )

        os.rename(item["old_path"], temp_path)

        temp_files.append({
            "temp_path": temp_path,
            "new_path": item["new_path"],
            "item": item,
        })

    # Stage 2:
    # Pindahkan semua file sementara ke tujuan akhir.
    for entry in temp_files:
        os.rename(entry["temp_path"], entry["new_path"])
        entry["item"]["status"] = "RENAMED"

except Exception:
    print(
        "\nERROR: Proses rename tidak selesai sepenuhnya."
    )
    print(
        "Jangan jalankan ulang sebelum memeriksa "
        "file sementara dan status setiap file."
    )

    for entry in temp_files:
        if os.path.exists(entry["temp_path"]):
            print(f'TEMP: {entry["temp_path"]}')

    save_report()
    raise

# ===================================
# SAVE FINAL REPORT
# ===================================

for item in label_mapping:
    if item["status"] == "SOURCE_MISSING_TARGET_EXISTS":
        # Status ini hanya menunjukkan sumber lama tidak ada
        # dan target ada; belum tentu membuktikan pasangan benar.
        continue

save_report()

# ===================================
# SUMMARY
# ===================================

print("\n=== RENAME LABEL SELESAI ===")

status_counts = Counter(
    item["status"] for item in label_mapping
)

for status, count in sorted(status_counts.items()):
    print(f"{status}: {count}")

print(
    f"\nLabel mapping CSV: "
    f"{os.path.abspath(LABEL_MAPPING_FILE)}"
)
