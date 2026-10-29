import os
import csv

# ===================================
# CONFIGURATION
# ===================================

IMAGE_ROOT = "dataset/raw2/images"
MAPPING_FILE = "results/dataset/rename_mapping.csv"

# ===================================
# VALIDATE
# ===================================

if not os.path.isdir(IMAGE_ROOT):
    raise FileNotFoundError(IMAGE_ROOT)

if not os.path.isfile(MAPPING_FILE):
    raise FileNotFoundError(MAPPING_FILE)

with open(
    MAPPING_FILE, "r", newline="", encoding="utf-8-sig"
) as file:
    mapping = list(csv.DictReader(file))

if not mapping:
    raise ValueError("Mapping CSV kosong. Hentikan pemulihan.")

# ===================================
# BUILD RECOVERY PLAN
# ===================================

items = []

for index, item in enumerate(mapping, start=1):
    folder = item["folder"]

    folder_path = os.path.join(IMAGE_ROOT, folder)

    old_path = os.path.join(folder_path, item["old_name"])
    new_path = os.path.join(folder_path, item["new_name"])

    # Sama persis dengan pola script rename awal.
    temp_path = os.path.join(
        folder_path, f"__temp_rename_{index:08d}__"
    )

    items.append({
        "index": index,
        "old_path": old_path,
        "temp_path": temp_path,
        "new_path": new_path,
    })

# ===================================
# STAGE 1: COMPLETE TEMPORARY RENAMES
# ===================================

print("Checking file states...")

for item in items:
    old_path = item["old_path"]
    temp_path = item["temp_path"]
    new_path = item["new_path"]

    if os.path.exists(temp_path):
        # File sudah berada di tahap temporary.
        continue

    if os.path.exists(old_path):
        # Tahap pertama sebelumnya belum mencapai file ini.
        os.rename(old_path, temp_path)
        continue

    if os.path.exists(new_path):
        # Kemungkinan file sudah selesai di-rename.
        continue

    raise FileNotFoundError(
        "File tidak ditemukan di ketiga lokasi:\n"
        f"Original: {old_path}\n"
        f"Temporary: {temp_path}\n"
        f"Final: {new_path}"
    )

# ===================================
# STAGE 2: CHECK CONFLICTS
# ===================================

conflicts = []

for item in items:
    if (
        os.path.exists(item["temp_path"])
        and os.path.exists(item["new_path"])
    ):
        conflicts.append(item["new_path"])

if conflicts:
    print("\nPEMULIHAN DIHENTIKAN: ditemukan konflik nama.")
    for path in conflicts[:20]:
        print(f"  {path}")

    print(
        "\nFile temporary tidak dihapus atau ditimpa. "
        "Periksa konflik sebelum melanjutkan."
    )
    raise FileExistsError(
        f"Ditemukan {len(conflicts)} konflik."
    )

# ===================================
# STAGE 3: APPLY FINAL NAMES
# ===================================

recovered = 0
already_final = 0

for item in items:
    temp_path = item["temp_path"]
    new_path = item["new_path"]

    if os.path.exists(temp_path):
        os.rename(temp_path, new_path)
        recovered += 1
    elif os.path.exists(new_path):
        already_final += 1

# ===================================
# SUMMARY
# ===================================

print("\nPemulihan selesai.")
print(f"Files renamed from temporary: {recovered}")
print(f"Files already in final state: {already_final}")
print(f"Total mapping entries: {len(items)}")