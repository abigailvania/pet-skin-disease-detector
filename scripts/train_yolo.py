
import os
import csv
import json
import shutil
import math
from pathlib import Path
from datetime import datetime

import torch
import yaml
import matplotlib.pyplot as plt

from ultralytics import YOLO


# ===================================
# CONFIGURATION
# ===================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "dataset" / "processed"
RAW_DIR = PROJECT_DIR / "dataset" / "raw"

IMAGE_DIR = RAW_DIR / "images"
YOLO_LABEL_DIR = RAW_DIR / "labels"

MODEL_DIR = PROJECT_DIR / "models" / "detection"
RESULTS_DIR = PROJECT_DIR / "results" / "detection"

DATASET_YAML = DATASET_DIR / "data.yaml"

MODEL_NAME = "yolov8n.pt"

EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
PATIENCE = 20
SEED = 42

DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 0 if os.name == "nt" else 4

CLASS_NAMES = {
    0: "lesion"
}

RUN_NAME = (
    "lesion_detection_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
)


# ===================================
# CREATE DIRECTORIES
# ===================================

DATASET_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ===================================
# FIND SPLIT CSV FILES
# ===================================

def find_split_csv(filename):
    matches = [
        path
        for path in DATASET_DIR.rglob("*")
        if path.is_file()
        and path.name.lower() == filename.lower()
    ]

    if not matches:
        raise FileNotFoundError(
            f"File {filename} tidak ditemukan di {DATASET_DIR}"
        )

    matches.sort(
        key=lambda path: (
            0 if path.parent.name.lower() in {"split", "splits"} else 1,
            len(path.parts)
        )
    )

    if len(matches) > 1:
        print(f"Peringatan: beberapa file {filename} ditemukan.")
        print(f"Menggunakan: {matches[0]}")

    return matches[0]


SPLIT_CSV = {
    "train": find_split_csv("train.csv"),
    "val": find_split_csv("val.csv"),
    "test": find_split_csv("test.csv")
}

SPLIT_LISTS = {
    split: DATASET_DIR / f"{split}.txt"
    for split in SPLIT_CSV
}


# ===================================
# RESOLVE PATH
# ===================================

def resolve_path(path_value):
    path_value = str(path_value).strip().strip('"')

    if not path_value:
        return None

    path = Path(path_value)

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_DIR / path).resolve()


# ===================================
# READ SPLIT CSV
# ===================================

def read_split_csv(csv_file):
    records = []

    with open(
        csv_file,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.reader(file)

        for row_number, row in enumerate(reader, start=1):
            if not row or not any(value.strip() for value in row):
                continue

            row = [value.strip() for value in row]

            if row[0].lower() in {
                "image", "image_path", "image_file",
                "filepath", "file_path"
            }:
                continue

            if len(row) < 2:
                raise ValueError(
                    f"Kolom CSV tidak lengkap: {csv_file}, "
                    f"baris {row_number}"
                )

            image_path = resolve_path(row[0])
            class_name = row[1].upper()

            if image_path is None or not image_path.is_file():
                raise FileNotFoundError(
                    f"Gambar tidak ditemukan: {image_path}"
                )

            if class_name not in {
                "DEMODICOSIS",
                "DERMATITIS",
                "DERMATOFITOSIS",
                "HEALTHY"
            }:
                raise ValueError(
                    f"Kelas tidak dikenal: {class_name}, "
                    f"baris {row_number}"
                )

            records.append({
                "image": image_path,
                "class_name": class_name
            })

    return records


# ===================================
# VALIDATE YOLO LABEL
# ===================================

def validate_yolo_label(label_path, class_name):
    if not label_path.is_file():
        if class_name == "HEALTHY":
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("", encoding="utf-8")
            return 0

        raise FileNotFoundError(
            f"Label YOLO tidak ditemukan: {label_path}"
        )

    object_count = 0

    with open(label_path, "r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) != 5:
                raise ValueError(
                    f"Format label harus class_id x_center y_center "
                    f"width height.\nFile: {label_path}\n"
                    f"Baris: {line_number}"
                )

            try:
                class_id, x, y, width, height = map(float, values)
            except ValueError as error:
                raise ValueError(
                    f"Nilai label bukan angka: {label_path}, "
                    f"baris {line_number}"
                ) from error

            if not all(
                math.isfinite(value)
                for value in (class_id, x, y, width, height)
            ):
                raise ValueError(
                    f"Label memiliki nilai tidak valid: {label_path}"
                )

            if class_id != 0:
                raise ValueError(
                    f"Class ID harus 0 (lesion): {label_path}"
                )

            if not (
                0 <= x <= 1
                and 0 <= y <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            ):
                raise ValueError(
                    f"Koordinat atau ukuran bounding box tidak valid: "
                    f"{label_path}, baris {line_number}"
                )

            if (
                x - width / 2 < -1e-6
                or x + width / 2 > 1 + 1e-6
                or y - height / 2 < -1e-6
                or y + height / 2 > 1 + 1e-6
            ):
                raise ValueError(
                    f"Bounding box melewati batas gambar: "
                    f"{label_path}, baris {line_number}"
                )

            object_count += 1

    return object_count


# ===================================
# PREPARE DATASET
# ===================================

def prepare_dataset():
    total_images = 0
    total_objects = 0
    total_empty_healthy = 0
    split_records = {}

    for split, csv_file in SPLIT_CSV.items():
        print("\n" + "=" * 50)
        print(f"Memproses split: {split.upper()}")
        print(f"CSV: {csv_file}")
        print("=" * 50)

        records = read_split_csv(csv_file)
        split_records[split] = records

        image_list = []

        for record in records:
            image_path = record["image"]
            class_name = record["class_name"]

            try:
                image_path.relative_to(IMAGE_DIR.resolve())
            except ValueError:
                raise ValueError(
                    f"Gambar harus berada di {IMAGE_DIR}: {image_path}"
                )

            # Gunakan label yang sudah dikonversi.
            # CSV kolom ketiga tidak digunakan.
            label_path = (
                YOLO_LABEL_DIR
                / class_name
                / f"{image_path.stem}.txt"
            )

            label_existed = label_path.is_file()

            object_count = validate_yolo_label(
                label_path,
                class_name
            )

            if class_name == "HEALTHY" and not label_existed:
                total_empty_healthy += 1

            total_objects += object_count
            total_images += 1

            # Ultralytics memetakan images/ ke labels/
            # sehingga label dibaca dari raw/labels/<kelas>/.
            image_list.append(str(image_path.resolve()))

        list_path = SPLIT_LISTS[split]

        with open(list_path, "w", encoding="utf-8") as file:
            for image_path in image_list:
                file.write(image_path + "\n")

        print(f"Jumlah gambar: {len(records)}")
        print(f"Daftar gambar: {list_path}")

    # Periksa agar gambar tidak muncul di beberapa split.
    image_sets = {
        split: {
            str(record["image"]).casefold()
            for record in records
        }
        for split, records in split_records.items()
    }

    split_names = list(image_sets)

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            split_a = split_names[i]
            split_b = split_names[j]

            overlap = image_sets[split_a] & image_sets[split_b]

            if overlap:
                raise ValueError(
                    f"Ada {len(overlap)} gambar yang muncul "
                    f"di split {split_a} dan {split_b}. "
                    f"Periksa CSV split."
                )

    print("\n" + "=" * 50)
    print("PERSIAPAN DATASET SELESAI")
    print("=" * 50)
    print(f"Total gambar: {total_images}")
    print(f"Total bounding box: {total_objects}")
    print(f"Label kosong HEALTHY yang dibuat: {total_empty_healthy}")


# ===================================
# CREATE DATA YAML
# ===================================

def create_data_yaml():
    data_config = {
        "path": str(PROJECT_DIR),
        "train": str(SPLIT_LISTS["train"].resolve()),
        "val": str(SPLIT_LISTS["val"].resolve()),
        "test": str(SPLIT_LISTS["test"].resolve()),
        "names": CLASS_NAMES
    }

    with open(DATASET_YAML, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            data_config,
            file,
            sort_keys=False,
            allow_unicode=True
        )

    print(f"\ndata.yaml disimpan: {DATASET_YAML}")
    print(yaml.safe_dump(data_config, sort_keys=False))


# ===================================
# TRAIN MODEL
# ===================================

def train_model():
    print("\n" + "=" * 50)
    print("MEMULAI TRAINING YOLO")
    print("=" * 50)

    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMAGE_SIZE}")

    model = YOLO(MODEL_NAME)

    model.train(
        data=str(DATASET_YAML.resolve()),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        patience=PATIENCE,
        seed=SEED,
        device=DEVICE,
        workers=WORKERS,
        project=str(RESULTS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True
    )

    save_dir = Path(model.trainer.save_dir).resolve()

    best_model = save_dir / "weights" / "best.pt"
    last_model = save_dir / "weights" / "last.pt"

    if not best_model.is_file():
        raise FileNotFoundError(
            f"Model best.pt tidak ditemukan: {best_model}"
        )

    destination_best = MODEL_DIR / "lesion_detection_best.pt"
    shutil.copy2(best_model, destination_best)

    print(f"\nModel terbaik: {destination_best}")

    if last_model.is_file():
        destination_last = MODEL_DIR / "lesion_detection_last.pt"
        shutil.copy2(last_model, destination_last)
        print(f"Model terakhir: {destination_last}")

    return destination_best, save_dir


# ===================================
# EVALUATE MODEL
# ===================================

def evaluate_model(model_path, save_dir):
    model = YOLO(str(model_path))
    evaluation_results = {}

    for split in ["val", "test"]:
        print(f"\nEvaluasi split: {split.upper()}")

        metrics = model.val(
            data=str(DATASET_YAML.resolve()),
            split=split,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            workers=WORKERS,
            plots=True,
            project=str(RESULTS_DIR),
            name=f"{RUN_NAME}_{split}_evaluation",
            exist_ok=True,
            verbose=True
        )

        split_metrics = {}

        for key, value in metrics.results_dict.items():
            try:
                split_metrics[key] = float(value)
            except (TypeError, ValueError):
                split_metrics[key] = str(value)

        evaluation_results[split] = split_metrics

        for key, value in split_metrics.items():
            print(f"{key}: {value}")

    report_path = save_dir / "evaluation_metrics.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            evaluation_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nLaporan evaluasi: {report_path}")


# ===================================
# PLOT TRAINING CURVES
# ===================================

def plot_training_curves(save_dir):
    results_csv = save_dir / "results.csv"

    if not results_csv.is_file():
        print("results.csv tidak ditemukan. Grafik dilewati.")
        return

    with open(
        results_csv,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        print("results.csv kosong.")
        return

    loss_columns = [
        column
        for column in rows[0]
        if "loss" in column.lower()
    ]

    if not loss_columns:
        print("Kolom loss tidak ditemukan.")
        return

    epochs = list(range(1, len(rows) + 1))

    for column in loss_columns:
        values = []

        for row in rows:
            try:
                values.append(float(row[column]))
            except (ValueError, TypeError):
                values.append(float("nan"))

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, values)
        plt.xlabel("Epoch")
        plt.ylabel(column)
        plt.title(f"Training Curve - {column}")
        plt.grid(True)
        plt.tight_layout()

        safe_name = (
            column.strip()
            .replace("/", "_")
            .replace(" ", "_")
        )

        plot_path = save_dir / f"{safe_name}.png"

        plt.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        print(f"Grafik disimpan: {plot_path}")


# ===================================
# MAIN
# ===================================

def main():
    print("=" * 50)
    print("YOLO LESION DETECTION")
    print("=" * 50)

    print(f"Project: {PROJECT_DIR}")
    print(f"Images: {IMAGE_DIR}")
    print(f"YOLO labels: {YOLO_LABEL_DIR}")

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Folder gambar tidak ditemukan: {IMAGE_DIR}"
        )

    if not YOLO_LABEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Folder label YOLO tidak ditemukan: {YOLO_LABEL_DIR}"
        )

    # 1. Gunakan label YOLO yang sudah tersedia.
    prepare_dataset()

    # 2. Buat konfigurasi dataset.
    create_data_yaml()

    # 3. Training.
    best_model, save_dir = train_model()

    # 4. Evaluasi validation dan test.
    evaluate_model(best_model, save_dir)

    # 5. Simpan grafik loss.
    plot_training_curves(save_dir)

    print("\n" + "=" * 50)
    print("SEMUA PROSES SELESAI")
    print("=" * 50)
    print(f"Model: {best_model}")
    print(f"Hasil: {save_dir}")
    print(f"Model tersimpan di: {MODEL_DIR}")


if __name__ == "__main__":
    main()
