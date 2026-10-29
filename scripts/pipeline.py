import os
import json
import cv2
import numpy as np
import tensorflow as tf

from PIL import Image
from ultralytics import YOLO

# ===================================
# CONFIGURATION
# ===================================

CLASSIFICATION_MODEL_PATH = "models/classification/best_model.keras"
CLASS_NAMES_PATH = "dataset/classification/class_names.json"
DETECTION_MODEL_PATH = "models/detection/best_model.pt"
SEGMENTATION_MODEL_PATH = "models/segmentation/best_unet.keras"

IMAGE_PATH = "test_images/test.jpg"
OUTPUT_DIR = "results/inference"

CLASSIFICATION_IMG_SIZE = (224, 224)
SEGMENTATION_IMG_SIZE = (256, 256)

CLASSIFICATION_THRESHOLD = 0.5
DETECTION_THRESHOLD = 0.25
SEGMENTATION_THRESHOLD = 0.5

# ===================================
# LOAD MODELS
# ===================================

classification_model = tf.keras.models.load_model(CLASSIFICATION_MODEL_PATH, compile=False)
detection_model = YOLO(DETECTION_MODEL_PATH)
segmentation_model = tf.keras.models.load_model(SEGMENTATION_MODEL_PATH, compile=False)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load()

# ===================================
# CLASSIFICATION
# ===================================

classification_image = Image.open(IMAGE_PATH).convert("RGB")
classification_image = classification_image.resize(CLASSIFICATION_IMG_SIZE)

classification_array = np.array(classification_image).astype("float32") / 255.0
classification_array = np.expand_dims(classification_array, axis=0)

classification_predictions = (classification_model.predict(classification_array, verbose=0)[0])

classification_probabilities = {}
detected_diseases = []

for class_name, probability in zip(class_names, classification_predictions):
    probability = float(probability)
    classification_probabilities[class_name] = (probability)

    if probability >= CLASSIFICATION_THRESHOLD:
        detected_diseases.append(class_names)

# ===================================
# DETECTION
# ===================================

detection_results = detection_model.predict(
    source=IMAGE_PATH,
    conf=DETECTION_THRESHOLD,
    imgsz=640,
    verbose=False,
    device=0
)

detection_result = detection_results[0]
detections = []

if detection_result.boxes is not None:
    for index, box in enumerate(detection_result.boxes):
        xyxy = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf[0].cpu().numpy())
        class_id = int(box.cls[0].cpu().numpy())
        x1, y1, x2, y2 = map(int, xyxy)
        detections.append({
            "lesion_id": index + 1,
            "class_id": class_id,
            "class_name": detection_model.names[class_id],
            "confidence": confidence,
            "bounding_box": { "x1": x1, "y1": y1, "x2": x2, "y2": y2 }
        })

# ===================================
# SEGMENTATION
# ===================================

segmentation_image = Image.open(IMAGE_PATH).convert("RGB")
segmentation_image = segmentation_image.resize(SEGMENTATION_IMG_SIZE)

segmentation_array = np.array(segmentation_image).astype("float32") / 255.0
segmentation_array = np.expand_dims(segmentation_array, axis=0)

segmentation_prediction = (segmentation_model.predict(segmentation_array, verbose=0)[0])

probability_mask = (segmentation_prediction[:, :, 0])

binary_mask = (probability_mask >= SEGMENTATION_THRESHOLD).astype(np.uint8)

total_pixels = binary_mask.size
lesion_pixels = int(np.sum(binary_mask > 0))
lesion_area_percentage = (lesion_pixels / total_pixels) * 100

# ===================================
# SEVERITY
# ===================================

def determine_severity(area_percentage):
    if area_percentage < 5:
        return "mild"
    elif area_percentage < 20:
        return "moderate"
    else:
        return "severe"

severity = determine_severity(lesion_area_percentage)

# ===================================
# RECOMMENDATION
# ===================================

def generate_recommendations(diseases, severity):
    recommendations = []

    if len(diseases) == 0:
        recommendations.append("No disease was detected above the confidence threshold")
        return recommendations

    if "ringworm" in diseases:
        recommendations.append("Keep the affected area clean and dry.")
        recommendations.append("Avoid sharing grooming tools with other pets.")

    if "dermatitis" in diseases:
        recommendations.append("Avoid potential skin irritants and monitor itching.")

    if "demodicosis" in diseases:
        recommendations.append("Consult a veterinarian for further examination and treatment.")

    if severity == "severe":
        recommendations.append("Veterinary consultation is strongly recommended.")
    elif severity == "moderate":
        recommendations.append("Schedule a veterinary consultation if symptoms persist or worsen.")
    else:
        recommendations.append("Monitor the affected area and consult a veterinarian if it worsens.")

    recommendations.append("This result is an AI-based preliminary screening and not a veterinary diagnosis.")

    return recommendations

recommendations = generate_recommendations(detected_diseases, severity)

# ===================================
# FINAL RESULT
# ===================================

final_result = {
    "image": IMAGE_PATH,
    "classification": {
        "probabiliies": classification_probabilities,
        "detected_diseases": detected_diseases,
        "threshold": CLASSIFICATION_THRESHOLD
    },
    "detection": {
        "number_of_lesions": len(detections),
        "detections": detections,
        "threshold": DETECTION_THRESHOLD
    },
    "segmentation": {
        "lesion_pixels": lesion_pixels,
        "total_pixels": total_pixels,
        "lesion_area_percentage": lesion_area_percentage,
        "threshold": SEGMENTATION_THRESHOLD,
    },
    "severity": severity,
    "recommendations": recommendations
}

output_path = os.path.join(OUTPUT_DIR, "inference_result.json")

with open(output_path, "w") as f:
    json.dump(final_result, f, indent=4)

# ===================================
# DISPLAY
# ===================================

print("Detected diseases:", detected_diseases)
print("Number of lesions:", len(detections))
print(f"Lesion area: " f"{lesion_area_percentage:.2f}%")
print("Severity:", severity)

print("\nRecommendations:")
for recommendation in recommendations:
    print(f"- {recommendation}")

print("\Saved to ")
print(output_path)