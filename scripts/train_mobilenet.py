
import os
import json
import csv

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ===================================
# CONFIGURATION
# ===================================

PROJECT_ROOT = os.path.abspath(".")
RAW_IMAGE_DIR = os.path.join(
    PROJECT_ROOT, "dataset", "raw", "images"
)

PROCESSED_DIR = os.path.join("dataset", "processed")
SPLIT_DIR = os.path.join(PROCESSED_DIR, "splits")

TRAIN_CSV = os.path.join(SPLIT_DIR, "train.csv")
VAL_CSV = os.path.join(SPLIT_DIR, "val.csv")
TEST_CSV = os.path.join(SPLIT_DIR, "test.csv")

CLASS_NAMES_FILE = os.path.join(
    PROCESSED_DIR, "class_names.json"
)

MODEL_DIR = os.path.join("models", "classification")
RESULTS_DIR = os.path.join("results", "classification")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Maksimal epoch untuk training awal.
EPOCHS = 50

# Maksimal epoch tambahan untuk fine-tuning.
FINE_TUNE_EPOCHS = 15

SEED = 42
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5

# Jumlah layer backbone yang tetap dibekukan.
FINE_TUNE_AT_RATIO = 0.70

# Early stopping.
EARLY_STOPPING_PATIENCE = 7

# ===================================
# REPRODUCIBILITY
# ===================================

tf.keras.utils.set_random_seed(SEED)

# ===================================
# CREATE DIRECTORIES
# ===================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR, "best_model.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR, "final_model.keras"
)

# ===================================
# CHECK FILE
# ===================================

required_files = [
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    CLASS_NAMES_FILE,
]

for file_path in required_files:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Required file not found: {os.path.abspath(file_path)}"
        )

if not os.path.isdir(RAW_IMAGE_DIR):
    raise FileNotFoundError(
        f"Raw image directory not found: {RAW_IMAGE_DIR}"
    )

# ===================================
# LOAD CLASS NAMES
# ===================================

with open(
    CLASS_NAMES_FILE, "r", encoding="utf-8-sig"
) as f:
    class_names = json.load(f)

if not isinstance(class_names, list) or not class_names:
    raise ValueError(
        "class_names.json harus berisi list nama kelas."
    )

if len(set(class_names)) != len(class_names):
    raise ValueError(
        "Terdapat nama kelas duplikat dalam class_names.json."
    )

num_classes = len(class_names)

print("Class names:", class_names)
print("Number of classes:", num_classes)

# ===================================
# LOAD CSV
# ===================================

def read_csv_dataset(csv_path):
    image_paths = []
    labels = []

    class_to_index = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    with open(
        csv_path, "r",
        newline="", encoding="utf-8-sig"
    ) as f:
        reader = csv.DictReader(f)

        required_columns = {"image", "label"}

        if not reader.fieldnames or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"CSV harus memiliki kolom: {required_columns}. "
                f"File: {csv_path}"
            )

        for row_number, row in enumerate(reader, start=2):
            image_name = (row.get("image") or "").strip()
            label_name = (row.get("label") or "").strip()

            if not image_name or not label_name:
                raise ValueError(
                    f"Path gambar atau label kosong pada "
                    f"{csv_path}, baris {row_number}."
                )

            # Path pada CSV relatif terhadap root project.
            if os.path.isabs(image_name):
                image_path = os.path.normpath(image_name)
            else:
                image_path = os.path.normpath(
                    os.path.join(PROJECT_ROOT, image_name)
                )

            if not os.path.isfile(image_path):
                raise FileNotFoundError(
                    f"Gambar tidak ditemukan: {image_path}"
                )

            # Pastikan gambar berasal dari dataset/raw/images.
            image_path_abs = os.path.normcase(
                os.path.abspath(image_path)
            )
            raw_image_dir_abs = os.path.normcase(
                os.path.abspath(RAW_IMAGE_DIR)
            )

            try:
                common_path = os.path.commonpath([
                    image_path_abs,
                    raw_image_dir_abs,
                ])
            except ValueError as error:
                raise ValueError(
                    f"Path gambar tidak valid: {image_path}"
                ) from error

            if common_path != raw_image_dir_abs:
                raise ValueError(
                    "Path gambar berada di luar dataset/raw/images: "
                    f"{image_path}"
                )

            if label_name not in class_to_index:
                raise ValueError(
                    f"Label tidak dikenal: {label_name}. "
                    f"File: {csv_path}, baris {row_number}"
                )

            image_paths.append(image_path)
            labels.append(class_to_index[label_name])

    if not image_paths:
        raise ValueError(
            f"Tidak ada data gambar dalam {csv_path}"
        )

    return (
        np.array(image_paths),
        np.array(labels, dtype=np.int32)
    )

# ===================================
# LOAD DATASET
# ===================================

train_paths, y_train = read_csv_dataset(TRAIN_CSV)
val_paths, y_val = read_csv_dataset(VAL_CSV)
test_paths, y_test = read_csv_dataset(TEST_CSV)

# ===================================
# CHECK DATA SPLIT
# ===================================

train_path_set = {
    os.path.normcase(os.path.abspath(path))
    for path in train_paths
}

val_path_set = {
    os.path.normcase(os.path.abspath(path))
    for path in val_paths
}

test_path_set = {
    os.path.normcase(os.path.abspath(path))
    for path in test_paths
}

if train_path_set & val_path_set:
    raise ValueError(
        "Terdapat gambar yang sama pada train dan validation."
    )

if train_path_set & test_path_set:
    raise ValueError(
        "Terdapat gambar yang sama pada train dan test."
    )

if val_path_set & test_path_set:
    raise ValueError(
        "Terdapat gambar yang sama pada validation dan test."
    )

print("\n=== DATASET SUMMARY ===")
print(f"Training images:   {len(train_paths)}")
print(f"Validation images: {len(val_paths)}")
print(f"Test images:       {len(test_paths)}")

for index, class_name in enumerate(class_names):
    print(
        f"{class_name}: "
        f"train={np.sum(y_train == index)}, "
        f"val={np.sum(y_val == index)}, "
        f"test={np.sum(y_test == index)}"
    )

# ===================================
# CREATE TF.DATA DATASET
# ===================================

def load_image(image_path, label):
    image = tf.io.read_file(image_path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)

    # Pixel tetap berada pada rentang 0-255.
    # MobileNetV3 include_preprocessing=True
    # menangani preprocessing internal.
    return image, label


def create_dataset(image_paths, labels, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_ds = create_dataset(
    train_paths, y_train, shuffle=True
)

val_ds = create_dataset(
    val_paths, y_val, shuffle=False
)

test_ds = create_dataset(
    test_paths, y_test, shuffle=False
)

# ===================================
# DATA AUGMENTATION
# ===================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ],
    name="data_augmentation",
)

# ===================================
# BASE MODEL
# ===================================

base_model = tf.keras.applications.MobileNetV3Small(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
    include_preprocessing=True,
)

base_model.trainable = False

# ===================================
# MODEL
# ===================================

inputs = tf.keras.Input(
    shape=IMG_SIZE + (3,)
)

x = data_augmentation(inputs)

# training=False menjaga BatchNormalization
# backbone menggunakan statistik inference.
x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)

x = tf.keras.layers.Dense(
    128,
    activation="relu"
)(x)

x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

# ===================================
# COMPILE MODEL
# ===================================

def compile_model(learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],
    )


compile_model(LEARNING_RATE)

# ===================================
# CALLBACKS
# ===================================

# Callback checkpoint yang sama dipakai pada
# kedua tahap agar best_model.keras menyimpan
# model dengan val_loss terbaik secara keseluruhan.
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=BEST_MODEL_PATH,
    save_best_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1
)

def create_callbacks():
    return [
        checkpoint_callback,

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

# ===================================
# TRAIN MODEL
# ===================================

print("\n=== STAGE 1: TRAIN CLASSIFICATION HEAD ===")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=create_callbacks(),
)

# ===================================
# FINE-TUNING
# ===================================

print("\n=== STAGE 2: FINE-TUNING ===")

# Mulai fine-tuning dari bobot terbaik
# yang sudah tersimpan pada tahap pertama.
if not os.path.isfile(BEST_MODEL_PATH):
    raise FileNotFoundError(
        "Best model tidak ditemukan setelah training tahap pertama."
    )

model = tf.keras.models.load_model(BEST_MODEL_PATH)

# Ambil backbone dari model yang telah dilatih.
base_model = model.get_layer("MobileNetV3Small")
base_model.trainable = True

# Bekukan sebagian besar layer awal.
FINE_TUNE_AT = int(
    len(base_model.layers) * FINE_TUNE_AT_RATIO
)

for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

# Bekukan semua BatchNormalization.
for layer in base_model.layers:
    if isinstance(
        layer, tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False

print(f"Total backbone layers: {len(base_model.layers)}")
print(f"Fine-tuning starts at layer: {FINE_TUNE_AT}")
print(
    "Trainable backbone layers:",
    sum(
        1 for layer in base_model.layers
        if layer.trainable
    )
)

# Recompile setelah mengubah trainable layers.
compile_model(FINE_TUNE_LEARNING_RATE)

fine_tune_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=create_callbacks(),
)

# ===================================
# COMBINE TRAINING HISTORY
# ===================================

combined_history = {}

for metric_name in history.history:
    combined_history[metric_name] = (
        history.history[metric_name]
        + fine_tune_history.history.get(metric_name, [])
    )

# ===================================
# LOAD BEST MODEL
# ===================================

best_model = tf.keras.models.load_model(
    BEST_MODEL_PATH
)

# Simpan final_model sebagai salinan
# dari model dengan validation loss terbaik.
best_model.save(FINAL_MODEL_PATH)

print(f"\nBest model: {os.path.abspath(BEST_MODEL_PATH)}")
print(f"Final model: {os.path.abspath(FINAL_MODEL_PATH)}")

# ===================================
# GRAPH TRAINING HISTORY
# ===================================

epochs_completed = len(
    combined_history["loss"]
)

epoch_numbers = range(1, epochs_completed + 1)

# Loss graph.
plt.figure(figsize=(10, 5))

plt.plot(
    epoch_numbers,
    combined_history["loss"],
    label="Training Loss"
)

plt.plot(
    epoch_numbers,
    combined_history["val_loss"],
    label="Validation Loss"
)

if len(history.history["loss"]) > 0:
    plt.axvline(
        x=len(history.history["loss"]) + 0.5,
        linestyle="--",
        label="Fine-tuning starts"
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MobileNetV3 Training and Validation Loss")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "loss.png"),
    dpi=300
)

plt.close()

# Accuracy graph.
plt.figure(figsize=(10, 5))

plt.plot(
    epoch_numbers,
    combined_history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    epoch_numbers,
    combined_history["val_accuracy"],
    label="Validation Accuracy"
)

if len(history.history["accuracy"]) > 0:
    plt.axvline(
        x=len(history.history["accuracy"]) + 0.5,
        linestyle="--",
        label="Fine-tuning starts"
    )

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MobileNetV3 Training and Validation Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "accuracy.png"),
    dpi=300
)

plt.close()

# ===================================
# SAVE TRAINING HISTORY
# ===================================

history_csv_path = os.path.join(
    RESULTS_DIR, "training_history.csv"
)

with open(
    history_csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.writer(f)

    metric_names = list(combined_history.keys())

    writer.writerow(["epoch"] + metric_names)

    for index in range(epochs_completed):
        writer.writerow(
            [index + 1]
            + [
                combined_history[name][index]
                for name in metric_names
            ]
        )

# ===================================
# EVALUATE MODEL
# ===================================

print("\n=== TEST SET EVALUATION ===")

# return_dict=True agar nama metrik
# tidak bergantung pada urutan metrics_names.
test_results = best_model.evaluate(
    test_ds,
    verbose=1,
    return_dict=True
)

for metric_name, value in test_results.items():
    print(f"{metric_name}: {value:.4f}")

# ===================================
# PREDICTIONS
# ===================================

y_prob = best_model.predict(
    test_ds,
    verbose=1
)

y_true = y_test
y_pred = np.argmax(y_prob, axis=1)

# ===================================
# METRICS
# ===================================

precision_macro = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

precision_weighted = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall_weighted = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n=== OVERALL METRICS ===")

print(f"Macro Precision:    {precision_macro:.4f}")
print(f"Macro Recall:       {recall_macro:.4f}")
print(f"Macro F1-score:     {f1_macro:.4f}")

print(f"Weighted Precision: {precision_weighted:.4f}")
print(f"Weighted Recall:    {recall_weighted:.4f}")
print(f"Weighted F1-score:  {f1_weighted:.4f}")

# ===================================
# ROC-AUC
# ===================================

auc_score = None

try:
    if len(np.unique(y_true)) < num_classes:
        raise ValueError(
            "Test set tidak mencakup semua kelas."
        )

    auc_score = roc_auc_score(
        y_true,
        y_prob,
        labels=np.arange(num_classes),
        multi_class="ovr",
        average="macro"
    )

    print(f"Macro ROC-AUC (OvR): {auc_score:.4f}")

except ValueError as error:
    print(f"ROC-AUC tidak dapat dihitung: {error}")

# ===================================
# CLASSIFICATION REPORT
# ===================================

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    target_names=class_names,
    zero_division=0,
    digits=4
)

print("\n=== CLASSIFICATION REPORT ===")
print(report)

with open(
    os.path.join(
        RESULTS_DIR, "classification_report.txt"
    ),
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

# ===================================
# PER-CLASS METRICS
# ===================================

per_class_precision = precision_score(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    average=None,
    zero_division=0
)

per_class_recall = recall_score(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    average=None,
    zero_division=0
)

per_class_f1 = f1_score(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    average=None,
    zero_division=0
)

per_class_report = []

for index, class_name in enumerate(class_names):
    line = (
        f"{class_name}\n"
        f"  Precision: {per_class_precision[index]:.4f}\n"
        f"  Recall   : {per_class_recall[index]:.4f}\n"
        f"  F1-score : {per_class_f1[index]:.4f}\n"
    )

    per_class_report.append(line)

per_class_report_text = "\n".join(per_class_report)

print("\n=== PER-CLASS METRICS ===")
print(per_class_report_text)

with open(
    os.path.join(
        RESULTS_DIR, "per_class_metrics.txt"
    ),
    "w",
    encoding="utf-8"
) as f:
    f.write(per_class_report_text)

# ===================================
# MULTICLASS CONFUSION MATRIX
# ===================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(num_classes)
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("MobileNetV3 Multiclass Confusion Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR, "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()

# ===================================
# SAVE PREDICTIONS
# ===================================

prediction_file = os.path.join(
    RESULTS_DIR, "test_predictions.csv"
)

with open(
    prediction_file,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:
    writer = csv.writer(f)

    writer.writerow(
        ["image", "true_label", "predicted_label"]
        + [
            f"{name}_probability"
            for name in class_names
        ]
    )

    for i, image_path in enumerate(test_paths):
        writer.writerow(
            [
                os.path.relpath(
                    image_path, PROJECT_ROOT
                ),
                class_names[y_true[i]],
                class_names[y_pred[i]],
            ]
            + [
                f"{probability:.6f}"
                for probability in y_prob[i]
            ]
        )

# ===================================
# SAVE EVALUATION
# ===================================

evaluation_path = os.path.join(
    RESULTS_DIR, "evaluation_results.txt"
)

with open(
    evaluation_path,
    "w",
    encoding="utf-8"
) as f:
    f.write("MobileNetV3 Evaluation Results\n")
    f.write("=" * 35 + "\n\n")

    f.write(f"Classes: {', '.join(class_names)}\n")
    f.write(f"Training epochs completed: {epochs_completed}\n")
    f.write(f"Stage 1 epochs: {len(history.history['loss'])}\n")
    f.write(
        f"Stage 2 epochs: "
        f"{len(fine_tune_history.history['loss'])}\n\n"
    )

    f.write(f"Test Loss: {test_results['loss']:.4f}\n")

    if "accuracy" in test_results:
        f.write(
            f"Test Accuracy: {test_results['accuracy']:.4f}\n\n"
        )

    f.write(f"Macro Precision: {precision_macro:.4f}\n")
    f.write(f"Macro Recall: {recall_macro:.4f}\n")
    f.write(f"Macro F1-score: {f1_macro:.4f}\n\n")

    f.write(
        f"Weighted Precision: {precision_weighted:.4f}\n"
    )
    f.write(
        f"Weighted Recall: {recall_weighted:.4f}\n"
    )
    f.write(
        f"Weighted F1-score: {f1_weighted:.4f}\n"
    )

    if auc_score is not None:
        f.write(
            f"\nMacro ROC-AUC (OvR): {auc_score:.4f}\n"
        )
    else:
        f.write(
            "\nMacro ROC-AUC (OvR): Not available\n"
        )

# ===================================
# SUMMARY
# ===================================

print("\n=== TRAINING COMPLETED ===")

print(
    f"Total epochs completed: {epochs_completed}"
)

print(
    f"Best model: {os.path.abspath(BEST_MODEL_PATH)}"
)

print(
    f"Final model: {os.path.abspath(FINAL_MODEL_PATH)}"
)

print(
    f"Results directory: {os.path.abspath(RESULTS_DIR)}"
)

print("\nGenerated files:")
print("- loss.png")
print("- accuracy.png")
print("- training_history.csv")
print("- classification_report.txt")
print("- per_class_metrics.txt")
print("- confusion_matrix.png")
print("- test_predictions.csv")
print("- evaluation_results.txt")
