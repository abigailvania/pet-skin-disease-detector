
import os
import csv
import math
from pathlib import Path
from datetime import datetime


# ===================================
# CONFIGURATION
# ===================================

PROJECT_DIR = Path.cwd().resolve()

RAW_DIR = PROJECT_DIR / "dataset" / "raw"
IMAGE_DIR = RAW_DIR / "images"
SOURCE_LABEL_DIR = RAW_DIR / "detection_labels"
OUTPUT_LABEL_DIR = RAW_DIR / "labels"

RESULTS_DIR = PROJECT_DIR / "results" / "dataset"

REPORT_FILE = (
    RESULTS_DIR
    / f"yolo_conversion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

TARGET_CLASS_ID = 0

# Koordinat dinormalisasi ke rentang 0 sampai 1.
COORDINATE_MIN = 0.0
COORDINATE_MAX = 1.0

# Peringatan untuk bounding box yang mencakup
# sebagian besar gambar.
LARGE_BOX_THRESHOLD = 0.90


# ===================================
# CONVERT BOUNDING BOX
# ===================================

def convert_bbox(coordinates):
    """
    Format input:
        x_center y_center width height

    Konversi ke koordinat sudut, batasi ke 0-1,
    kemudian hitung ulang bounding box.
    """

    x_center, y_center, width, height = coordinates

    if width <= 0 or height <= 0:
        raise ValueError(
            "Lebar dan tinggi bounding box harus positif."
        )

    x_min = x_center - width / 2
    y_min = y_center - height / 2

    x_max = x_center + width / 2
    y_max = y_center + height / 2

    original = [x_min, y_min, x_max, y_max]

    clipped = [
        min(COORDINATE_MAX, max(COORDINATE_MIN, value))
        for value in original
    ]

    x_min, y_min, x_max, y_max = clipped

    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "Bounding box menjadi kosong setelah clipping."
        )

    new_x_center = (x_min + x_max) / 2
    new_y_center = (y_min + y_max) / 2

    new_width = x_max - x_min
    new_height = y_max - y_min

    changed = any(
        not math.isclose(a, b, abs_tol=1e-12)
        for a, b in zip(original, clipped)
    )

    return (
        new_x_center,
        new_y_center,
        new_width,
        new_height,
        changed
    )


# ===================================
# CONVERT POLYGON
# ===================================

def convert_polygon(coordinates):
    """
    Format input:
        x1 y1 x2 y2 x3 y3 ...

    Polygon dikonversi menjadi bounding box YOLO detection.
    Setiap titik dibatasi ke rentang 0-1 sebelum bbox dihitung.
    """

    if len(coordinates) < 6:
        raise ValueError(
            "Polygon harus memiliki minimal 3 titik."
        )

    if len(coordinates) % 2 != 0:
        raise ValueError(
            "Jumlah koordinat polygon harus genap."
        )

    x_coordinates = coordinates[0::2]
    y_coordinates = coordinates[1::2]

    clipped_x = [
        min(COORDINATE_MAX, max(COORDINATE_MIN, value))
        for value in x_coordinates
    ]

    clipped_y = [
        min(COORDINATE_MAX, max(COORDINATE_MIN, value))
        for value in y_coordinates
    ]

    changed = any(
        not math.isclose(a, b, abs_tol=1e-12)
        for a, b in zip(x_coordinates, clipped_x)
    ) or any(
        not math.isclose(a, b, abs_tol=1e-12)
        for a, b in zip(y_coordinates, clipped_y)
    )

    x_min = min(clipped_x)
    x_max = max(clipped_x)

    y_min = min(clipped_y)
    y_max = max(clipped_y)

    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "Polygon menjadi kosong setelah clipping."
        )

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    width = x_max - x_min
    height = y_max - y_min

    return (
        x_center,
        y_center,
        width,
        height,
        changed
    )


# ===================================
# CONVERT ONE LABEL FILE
# ===================================

def convert_label_file(source_file, output_file):
    """
    Menerima:
        YOLO bbox:
            class_id x_center y_center width height

        YOLO polygon:
            class_id x1 y1 x2 y2 x3 y3 ...

    Semua class_id sumber yang valid dipetakan
    menjadi TARGET_CLASS_ID.
    """

    converted_lines = []
    clipped_count = 0
    large_box_count = 0

    with open(
        source_file,
        "r",
        encoding="utf-8-sig"
    ) as file:
        lines = file.readlines()

    for line_number, line in enumerate(lines, start=1):
        line = line.strip()

        if not line:
            continue

        values = line.split()

        try:
            numbers = [float(value) for value in values]
        except ValueError:
            raise ValueError(
                f"Nilai bukan angka pada baris {line_number}."
            )

        if len(numbers) < 5:
            raise ValueError(
                f"Jumlah nilai tidak cukup pada baris {line_number}."
            )

        class_id = numbers[0]
        coordinates = numbers[1:]

        if (
            not math.isfinite(class_id)
            or not class_id.is_integer()
            or class_id < 0
        ):
            raise ValueError(
                f"class_id tidak valid pada baris {line_number}."
            )

        if not all(
            math.isfinite(value)
            for value in coordinates
        ):
            raise ValueError(
                f"Koordinat tidak valid pada baris {line_number}."
            )

        # --------------------------------
        # YOLO BOUNDING BOX
        # --------------------------------

        if len(coordinates) == 4:
            (
                x_center,
                y_center,
                width,
                height,
                changed
            ) = convert_bbox(coordinates)

        # --------------------------------
        # YOLO POLYGON
        # --------------------------------

        else:
            (
                x_center,
                y_center,
                width,
                height,
                changed
            ) = convert_polygon(coordinates)

        if changed:
            clipped_count += 1

        if (
            width >= LARGE_BOX_THRESHOLD
            or height >= LARGE_BOX_THRESHOLD
        ):
            large_box_count += 1

        # --------------------------------
        # FINAL VALIDATION
        # --------------------------------

        if not (
            0 <= x_center <= 1
            and 0 <= y_center <= 1
            and 0 < width <= 1
            and 0 < height <= 1
        ):
            raise ValueError(
                f"Bounding box tidak valid pada baris {line_number}."
            )

        converted_lines.append(
            f"{TARGET_CLASS_ID} "
            f"{x_center:.8f} "
            f"{y_center:.8f} "
            f"{width:.8f} "
            f"{height:.8f}\n"
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:
        file.writelines(converted_lines)

    return {
        "lines": len(converted_lines),
        "clipped": clipped_count,
        "large_boxes": large_box_count
    }


# ===================================
# CREATE EMPTY HEALTHY LABELS
# ===================================

def create_healthy_labels():
    """
    Setiap gambar HEALTHY harus memiliki label kosong.
    Label kosong menandakan tidak ada objek lesion.
    """

    healthy_dir = IMAGE_DIR / "HEALTHY"

    if not healthy_dir.is_dir():
        print(
            "Peringatan: folder HEALTHY tidak ditemukan."
        )
        return 0

    created = 0
    existing = 0

    for image_file in healthy_dir.rglob("*"):
        if (
            not image_file.is_file()
            or image_file.suffix.lower() not in IMAGE_EXTENSIONS
        ):
            continue

        output_file = (
            OUTPUT_LABEL_DIR
            / "HEALTHY"
            / f"{image_file.stem}.txt"
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Buat file kosong jika belum tersedia.
        # Jangan mengosongkan file yang sudah ada secara diam-diam.
        if not output_file.exists():
            output_file.write_text(
                "",
                encoding="utf-8"
            )
            created += 1
        else:
            existing += 1

    print(f"Label kosong HEALTHY dibuat: {created}")
    print(f"Label HEALTHY sudah tersedia: {existing}")

    return created


# ===================================
# CONVERT ALL LABELS
# ===================================

def convert_all_labels():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_LABEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not SOURCE_LABEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Folder label sumber tidak ditemukan: "
            f"{SOURCE_LABEL_DIR}"
        )

    report_rows = []

    total_files = 0
    total_success = 0
    total_errors = 0
    total_clipped = 0
    total_large_boxes = 0

    label_files = sorted(
        SOURCE_LABEL_DIR.rglob("*.txt")
    )

    print("=" * 60)
    print("YOLO LABEL CONVERTER")
    print("=" * 60)
    print(f"Label sumber: {SOURCE_LABEL_DIR}")
    print(f"Folder output: {OUTPUT_LABEL_DIR}")
    print(f"Jumlah file sumber: {len(label_files)}")

    for index, source_file in enumerate(
        label_files,
        start=1
    ):
        total_files += 1

        relative_path = source_file.relative_to(
            SOURCE_LABEL_DIR
        )

        output_file = OUTPUT_LABEL_DIR / relative_path

        # HEALTHY selalu dianggap tidak memiliki lesion.
        if relative_path.parts[0].upper() == "HEALTHY":
            output_file.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            output_file.write_text(
                "",
                encoding="utf-8"
            )

            total_success += 1

            report_rows.append({
                "source_file": str(source_file),
                "output_file": str(output_file),
                "status": "OK_HEALTHY_EMPTY",
                "objects": 0,
                "clipped_objects": 0,
                "large_boxes": 0,
                "message": "Label HEALTHY dikosongkan"
            })

            continue

        try:
            result = convert_label_file(
                source_file,
                output_file
            )

            total_success += 1
            total_clipped += result["clipped"]
            total_large_boxes += result["large_boxes"]

            if result["clipped"] > 0:
                status = "WARNING_CLIPPED"
                message = (
                    "Ada koordinat yang dibatasi ke rentang 0-1"
                )
            else:
                status = "OK"
                message = ""

            if result["large_boxes"] > 0:
                if message:
                    message += "; "

                message += (
                    "Ada bounding box yang mencakup "
                    "sebagian besar gambar; periksa anotasi"
                )

                if status == "OK":
                    status = "WARNING_LARGE_BOX"

            report_rows.append({
                "source_file": str(source_file),
                "output_file": str(output_file),
                "status": status,
                "objects": result["lines"],
                "clipped_objects": result["clipped"],
                "large_boxes": result["large_boxes"],
                "message": message
            })

        except Exception as error:
            total_errors += 1

            # Hapus output lama agar file gagal tidak
            # meninggalkan label hasil konversi sebelumnya.
            if output_file.exists():
                output_file.unlink()

            report_rows.append({
                "source_file": str(source_file),
                "output_file": str(output_file),
                "status": "ERROR",
                "objects": 0,
                "clipped_objects": 0,
                "large_boxes": 0,
                "message": str(error)
            })

        if index % 100 == 0 or index == len(label_files):
            print(
                f"Progress: {index}/{len(label_files)}"
            )

    # Buat label kosong untuk gambar HEALTHY
    # yang mungkin tidak mempunyai file sumber .txt.
    create_healthy_labels()

    # --------------------------------
    # SAVE REPORT
    # --------------------------------

    fieldnames = [
        "source_file",
        "output_file",
        "status",
        "objects",
        "clipped_objects",
        "large_boxes",
        "message"
    ]

    with open(
        REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(report_rows)

    print("\n" + "=" * 60)
    print("KONVERSI SELESAI")
    print("=" * 60)
    print(f"Total file sumber: {total_files}")
    print(f"Berhasil: {total_success}")
    print(f"Gagal: {total_errors}")
    print(f"Objek dengan koordinat di-clip: {total_clipped}")
    print(f"Objek dengan bounding box besar: {total_large_boxes}")
    print(f"Laporan: {REPORT_FILE}")

    if total_errors > 0:
        print(
            "\nPeriksa baris berstatus ERROR pada laporan "
            "sebelum menjalankan training."
        )


# ===================================
# MAIN
# ===================================

def main():
    print(f"Project directory: {PROJECT_DIR}")

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Folder gambar tidak ditemukan: {IMAGE_DIR}"
        )

    convert_all_labels()


if __name__ == "__main__":
    main()
