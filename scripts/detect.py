import os
import json
import cv2
from ultralytics import YOLO

# ===================================
# CONFIGURATION
# ===================================

MODEL_PATH = "models/detection/best_model.pt"
IMAGE_PATH = "test_images/test.jpg"
OUTPUT_DIR = "results/detection/predictions"

IMAGE_SIZE = 640

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================================
# LOAD MODEL
# ===================================

model = YOLO(MODEL_PATH)

results = model.predict(
    source=IMAGE_PATH,
    conf=0.25,
    imgsz=IMAGE_SIZE,
    save=False,
    verbose=False
)

result = results[0]

# ===================================
# READ IMAGE
# ===================================

image = cv2.imread(IMAGE_PATH)

detections = []

# ===================================
# BOUNDING BOXES
# ===================================

if result.boxes is not None:
    boxes = result.boxes

    for index, box in enumerate(boxes):
        xyxy = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf[0].cpu().numpy())
        class_id = int(box.cls[0].cpu().numpy())

        x1, y1, x2, y2 = map(int, xyxy)

        class_name = model.names[class_id]

        detections.append({
            "lesion_id": index + 1,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "bounding_box": { "x1": x1, "y1": y1, "x2": x2, "y2": y2 }
        })

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = (f"{class_name} "f"{confidence * 100:.2f}%")

        cv2.putText(image, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# ===================================
# SAVE VISUAL
# ===================================

output_image_path = os.path.join(OUTPUT_DIR, "detected_result.jpg")
cv2.imwrite(output_image_path, image)

# ===================================
# SAVE JSON
# ===================================

output_json_path = os.path.join(OUTPUT_DIR, "detection_result.jpg")
detection_result = {
    "image": IMAGE_PATH,
    "confidence_threshold": 0.25,
    "number_of_lesions": len(detections),
    "detections": detections
}

with open(output_json_path, "w") as f:
    json.dump(detection_result, f, indent=4)

# ===================================
# DISPLAY RESULT
# ===================================

for detection in detections:
    print(
        f"Lesion {detection['class_name']} - "
        f"{detection['confidence'] * 100:.2f}%"
    )

print(f"\nImage: ")
print(output_image_path)

print(f"\nJSON: ")
print(output_json_path)