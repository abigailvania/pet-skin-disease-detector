import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "models/classification/best_model.keras"
CLASS_NAMES_PATH = "dataset/classification/class_names.json"
IMAGE_PATH = "test_images/test.jpg"

IMG_SIZE = (224, 224)

# ===================================
# LOAD MODEL AND CLASS NAMES
# ===================================

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# ===================================
# LOAD IMAGE
# ===================================

image = Image.open(IMAGE_PATH).convert("RGB")
image = image.resize(IMG_SIZE)

image_array = np.array(image).astype("float32") / 255.0
image_array = np.expand_dims(image_array, axis=0)

# ===================================
# PREDICT
# ===================================

predictions = model.predict(image_array, verbose=0)[0]
prediction_results = {}

for class_name, probability in zip(class_names, predictions):
    prediction_results[class_name] = float(probability)

# ===================================
# SHOW RESULTS
# ===================================

detected_diseases = []

for disease, probability in prediction_results.items():
    percentage = probability * 100

    print(f"{disease}: "f"{percentage:.2f}%")

    if probability >= 0.5:
        detected_diseases.append(disease)

if len(detected_diseases) == 0:
    print("No disease detected")
else: 
    for disease in detected_diseases:
        print(f"- {disease}")

# ===================================
# SAVE JSON RESULT
# ===================================

output_path = "results/classification/prediction_result.json"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

result = {
    "image": IMAGE_PATH,
    "threshold": 0.5,
    "probabilities": prediction_results,
    "detected_diseases": detected_diseases
}

with open(output_path, "w") as f:
    json.dump(result, f, indent=4)