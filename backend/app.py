from flask import Flask, render_template, request
import os
import pickle
import numpy as np
import json
import sys
from PIL import Image

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if sys.version_info[0] < 3:
    raise RuntimeError(
        "This app requires Python 3. Run it with a Python 3 interpreter to load the trained model."
    )

MODEL_PATH = os.path.join(BASE_DIR, "train_model.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

model = pickle.load(open(MODEL_PATH, "rb"))
label_encoder = pickle.load(open(LABEL_ENCODER_PATH, "rb"))


def prepare_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((32, 32))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.flatten().reshape(1, -1)
    return img_array


def get_accuracy():
    try:
        with open(METRICS_PATH, "r") as f:
            data = json.load(f)
            return round(data["accuracy"] * 100, 2)
    except:
        return None


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    result = None
    image_file = None
    accuracy = None

    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]

            if file.filename != "":
                filename = file.filename
                image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(image_path)

                processed_image = prepare_image(image_path)
                pred = model.predict(processed_image)[0]

                result = label_encoder.inverse_transform([pred])[0]
                if result.lower() == "healthy":
                    result = "Healthy Leaf"
                elif result.lower() == "diseased":
                    result = "Diseased Leaf"

                image_file = filename
                accuracy = get_accuracy()

    return render_template(
        "prediction.html",
        result=result,
        image_file=image_file,
        accuracy=accuracy
    )


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
