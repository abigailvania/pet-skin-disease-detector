import os
import json
import numpy as np
import tensorflow as tf
import cv2

from PIL import Image

# ===================================
# CONFIGURATION
# ===================================

MODEL_PATH = "models/segmentation/best_unet.keras"
IMAGE_PATH = "test_images/test.jpg"
OUTPUT_DIR = "results/segmentation/predictions"

IMG_SIZE = (256, 256)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================================
# LOAD MODEL AND IMAGE
# ===================================

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

original_image = Image.open(IMAGE_PATH).convert("RGB")
resized_image = original_image.resize(IMG_SIZE)

image_array = np.array(resized_image).astype("float32") / 255.0
image_array = np.expand_dims(image_array, axis=0)

# ===================================
# PREDICT MASK
# ===================================

prediction = model.predict(image_array, verbose=0)[0]
probability_mask = prediction[:, :, 0]
binary_mask = (probability_mask >= 0.5).astype(np.uint8) * 255

# ===================================
# SAVE MASK
# ===================================

mask_path = os.path.join(OUTPUT_DIR, "predicted_mask.png")
cv2.imwrite(mask_path, binary_mask)

# ===================================
# ADD OVERLAY
# ===================================

resized_original = np.array(resized_image)
overlay = resized_image.copy()

overlay[binary_mask > 0] = (
    overlay[binary_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
).astype(np.uint8)

overlay_path = os.path.join(OUTPUT_DIR, "segmentation_overlay.jpg")

cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

# ===================================
# CALCULATE AREA
# ===================================

total_pixels = binary_mask.shape[0] * binary_mask.shape[1]

lesion_pixels = np.sum(binary_mask > 0)
lesion_area_percentage = (lesion_pixels / total_pixels) * 100

# ===================================
# SAVE JSON
# ===================================

segmentation_result = {
    "image": IMAGE_PATH,
    "threshold": 0.5,
    "lesion_pixels": int(lesion_pixels),
    "total_pixels": int(total_pixels),
    "lesion_area_percentage": float(lesion_area_percentage),
    "mask_path": mask_path,
    "overlay_path": overlay_path
}

json_path = os.path.join(OUTPUT_DIR, "segmentation_result.json")

with open(json_path, "w") as f:
    json.dump(segmentation_result, f, indent=4)

# ===================================
# RESULTS
# ===================================

print(f"Lesion pixels: {lesion_pixels}")
print(f"Total pixels: {total_pixels}")
print(f"Lesion area: "f"{lesion_area_percentage:.2f}%")

print(f"\nMask:")
print(mask_path)

print("\nOverlay:")
print(overlay_path)

print("\nJSON:")
print(json_path)