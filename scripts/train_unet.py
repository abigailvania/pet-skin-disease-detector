import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, jaccard_score)

# ===================================
# CONFIGURATION
# ===================================

SEGMENTATION_DIR = "dataset/segmentation"
IMAGE_DIR = os.path.join(SEGMENTATION_DIR, "images")
MASK_DIR = os.path.join(SEGMENTATION_DIR, "masks")

MODEL_DIR = os.path.join("models", "segmentation", "best_unet.keras")
RESULTS_DIR = os.path.join("results", "segmentation")

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 30

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===================================
# LOAD DATASET
# ===================================

def load_dataset(split):
    image_dir = os.path.join(IMAGE_DIR, split)
    mask_dir = os.path.join(MASK_DIR, split)

    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Image or mask directory for '{split}' split does not exist.")

    image_files = sorted([
        f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if len(image_files) == 0:
        raise ValueError(f"No image files found in '{image_dir}'.")

    images = []
    masks = []
    missing_masks = []

    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, os.path.splitext(filename)[0] + ".png")

        if not os.path.exists(mask_path):
            missing_masks.append(filename)
            continue

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.float32) / 255.0
        mask = np.where(mask >= 0.5, 1.0, 0.0)
        mask = np.expand_dims(mask, axis=-1)

        images.append(img)
        masks.append(mask)

        if missing_masks:
            print(f"Warning: Missing masks for the following images in '{split}' split: {missing_masks}")

        if len(images) == 0:
            raise ValueError(f"No valid image-mask pairs found in '{split}' split.")

    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)

# ===================================
# LOAD TRAIN, VAL, TEST DATASETS
# ===================================

x_train, y_train = load_dataset("train")
x_val, y_val = load_dataset("val")
x_test, y_test = load_dataset("test")

print(f"Train dataset: {x_train.shape}, {y_train.shape}")
print(f"Validation dataset: {x_val.shape}, {y_val.shape}")
print(f"Test dataset: {x_test.shape}, {y_test.shape}")

# ===================================
# BUILD U-NET MODEL
# ===================================

def conv_block(x, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x

def build_unet():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    c1 = conv_block(inputs, 64)
    p1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = conv_block(p1, 128)
    p2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = conv_block(p2, 256)
    p3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c3)

    c4 = conv_block(p3, 512)

    u5 = tf.keras.layers.upSampling2D(size=(2, 2))(c4)
    u5 = tf.keras.layers.concatenate([u5, c3])
    c5 = conv_block(u5, 256)

    u6 = tf.keras.layers.upSampling2D(size=(2, 2))(c5)
    u6 = tf.keras.layers.concatenate([u6, c2])
    c6 = conv_block(u6, 128)

    u7 = tf.keras.layers.upSampling2D(size=(2, 2))(c6)
    u7 = tf.keras.layers.concatenate([u7, c1])
    c7 = conv_block(u7, 64)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(c7)

    return tf.keras.Model(inputs, outputs)

model = build_unet()
model.summary()

# ===================================
# DICE COEFFICIENT METRIC, DICE LOSS, AND COMBINED LOSS
# ===================================

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    denominator = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = (2. * intersection + smooth) / (denominator + smooth)
    return tf.reduce_mean(dice)

def dice_loss(y_true, y_pred, smooth=1e-6):
    return 1 - dice_coefficient(y_true, y_pred, smooth)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

# ===================================
# COMPILE MODEL
# ===================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ===================================
# CALLBACKS
# ===================================

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "best_unet.keras"),
        save_best_only=True,
        monitor="val_dice_coefficient",
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_dice_coefficient",
        mode="max",
        verbose=1,
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_dice_coefficient",
        mode="max",
        factor=0.5,
        patience=4,
        verbose=1
    )
]

# ===================================
# TRAIN MODEL
# ===================================

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=True
)

model.save(os.path.join(MODEL_DIR, "final_unet.keras"))

# ===================================
# LOAD BEST MODEL AND EVALUATE ON TEST SET
# ===================================

best_model_path = os.path.join(MODEL_DIR, "best_unet.keras")

if not os.path.exists(best_model_path):
    raise FileNotFoundError(f"Best model file '{best_model_path}' does not exist.")

best_model = tf.keras.models.load_model(best_model_path, custom_objects={
    "dice_coefficient": dice_coefficient,
    "combined_loss": combined_loss
})

test_results = best_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=1)

for metric_name, metric_value in zip(best_model.metrics_names, test_results):
    print(f"{metric_name}: {metric_value:.4f}")

# ===================================
# PREDICTION
# ===================================

y_prob = best_model.predict(x_test, batch_size=BATCH_SIZE, verbose=1)

y_pred = (y_prob >= 0.5).astype(np.uint8)

y_true = (y_test >= 0.5).astype(np.uint8)

# ===================================
# CALCULATE METRICS
# ===================================

y_true_flat = y_true.flatten()
y_pred_flat = y_pred.flatten()

accuracy = accuracy_score(y_true_flat, y_pred_flat)
precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
iou = jaccard_score(y_true_flat, y_pred_flat, zero_division=0)

dice = ( 2 * precision * recall / 2 (precision + recall)
        if precision + recall > 0
        else 0.0)

# ===================================
# PRINT METRICS
# ===================================

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Dice:      {dice:.4f}")
print(f"IoU:       {iou:.4f}")

# ===================================
# SAVE EVALUATION
# ===================================

evaluation_path = os.path.join(RESULTS_DIR, "evaluation_results.txt")

with open(evaluation_path, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"Dice: {dice:.4f}\n")
    f.write(f"IoU: {iou:.4f}\n")

# ===================================
# SAVE SAMPLE
# ===================================

sample_count = min(5, len(x_test))

for index in range(sample_count):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(x_test[index])
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(y_true[index].squeeze(), cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(y_pred[index].squeeze(), cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, f"prediction_{index + 1}.png")

    plt.savefig(output_path)
    plt.close()