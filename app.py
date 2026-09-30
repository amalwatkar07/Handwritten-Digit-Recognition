from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

from PIL import Image
import numpy as np

import os
import base64
import io


app = Flask(__name__)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = load_model("model/digit_model.keras")


# --------------------------------------------------
# Upload folder
# --------------------------------------------------

UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def preprocess_image(image):

    # Convert to grayscale
    image = image.convert("L")

    # Convert image to NumPy array
    image_array = np.array(image)

    # Invert if background is mostly white
    if image_array.mean() > 127:
        image_array = 255 - image_array

    # Remove very light pixels
    image_array[image_array < 50] = 0

    # Find digit pixels
    coords = np.argwhere(image_array > 0)

    # If nothing was drawn
    if coords.size == 0:
        return None

    # Find bounding box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the digit
    cropped = image_array[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    # Convert to PIL image
    cropped_image = Image.fromarray(cropped)

    # Keep aspect ratio
    width, height = cropped_image.size

    max_dimension = max(width, height)

    scale = 20 / max_dimension

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    cropped_image = cropped_image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # Create 28x28 black image
    final_image = Image.new(
        "L",
        (28, 28),
        0
    )

    # Center the digit
    x_position = (28 - new_width) // 2
    y_position = (28 - new_height) // 2

    final_image.paste(
        cropped_image,
        (x_position, y_position)
    )

    # Convert to NumPy
    final_array = np.array(final_image)

    # Normalize
    final_array = final_array / 255.0

    # Reshape for CNN
    final_array = final_array.reshape(
        1, 28, 28, 1
    )

    return final_array


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    confidence = None
    image_path = None


    # --------------------------------------------------
    # Uploaded image prediction
    # --------------------------------------------------

    if request.method == "POST":

        if "image" not in request.files:

            return render_template(
                "index.html",
                prediction=None,
                confidence=None,
                image_path=None
            )


        file = request.files["image"]


        if file.filename == "":
            return render_template(
                "index.html",
                prediction=None,
                confidence=None,
                image_path=None
            )


        # Save uploaded image
        filename = file.filename

        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(filepath)

        image_path = filepath.replace("\\", "/")


        # Open image
        image = Image.open(filepath)


        # Preprocess
        processed_image = preprocess_image(image)


        if processed_image is not None:

            result = model.predict(
                processed_image,
                verbose=0
            )

            prediction = int(
                np.argmax(result)
            )

            confidence = round(
                float(np.max(result)) * 100,
                2
            )


    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )


# --------------------------------------------------
# Canvas prediction
# --------------------------------------------------

@app.route("/predict_canvas", methods=["POST"])
def predict_canvas():

    try:

        data = request.get_json()


        if not data or "image" not in data:

            return jsonify({
                "error": "No drawing received."
            }), 400


        # Get Base64 image
        image_data = data["image"]


        # Remove Base64 header
        image_data = image_data.split(",")[1]


        # Decode image
        image_bytes = base64.b64decode(
            image_data
        )


        # Open image
        image = Image.open(
            io.BytesIO(image_bytes)
        )


        # Preprocess
        processed_image = preprocess_image(image)


        if processed_image is None:

            return jsonify({
                "error": "Please draw a digit first."
            }), 400


        # Prediction
        result = model.predict(
            processed_image,
            verbose=0
        )


        prediction = int(
            np.argmax(result)
        )


        confidence = round(
            float(np.max(result)) * 100,
            2
        )


        return jsonify({

            "prediction": prediction,

            "confidence": confidence

        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------------------------
# Run Flask
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )