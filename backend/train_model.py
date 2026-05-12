import os
import json
import pickle
import sys
import numpy as np
from PIL import Image

if sys.version_info[0] < 3:
    raise RuntimeError(
        "This training script requires Python 3. Run it with a Python 3 interpreter."
    )

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = r"D:\pd\train"

X = []
y = []

image_extensions = (".jpg", ".jpeg", ".png")

if not os.path.exists(dataset_path):
    print("Dataset path not found:", dataset_path)
    sys.exit(1)

class_folders = os.listdir(dataset_path)
print("Classes found:", class_folders)

for class_name in class_folders:
    class_folder = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_folder):
        print("Reading folder: {}".format(class_name))

        for image_name in os.listdir(class_folder):
            image_path = os.path.join(class_folder, image_name)

            if image_name.lower().endswith(image_extensions):
                try:
                    img = Image.open(image_path).convert("RGB")
                    img = img.resize((128, 128))
                    img_array = np.array(img) / 255.0
                    X.append(img_array.flatten())
                    y.append(class_name)
                except Exception as e:
                    print("Error in image:", image_path)
                    print("Reason:", e)

print("Total images loaded:", len(X))

if len(X) == 0:
    print("No images loaded. Please check dataset_path.")
    sys.exit(1)

X = np.array(X)
y = np.array(y)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)


with open(os.path.join(BASE_DIR, "train_model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)


metrics_data = {
    "accuracy": float(accuracy),
    "total_images": int(len(X)),
    "train_images": int(len(X_train)),
    "test_images": int(len(X_test)),
    "classes": list(label_encoder.classes_)
}

with open(os.path.join(BASE_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_data, f, indent=4)


monitoring_data = {
    "model_status": "trained",
    "accuracy": float(accuracy),
    "message": "Model trained successfully and ready for prediction"
}

with open(os.path.join(BASE_DIR, "monitoring.json"), "w") as f:
    json.dump(monitoring_data, f, indent=4)

print("Model, label encoder, metrics.json, and monitoring.json saved successfully.")
