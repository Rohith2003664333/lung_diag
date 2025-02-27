import os
import numpy as np
import gdown
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

# Google Drive model link (Replace with your File ID)
MODEL_PATH = "xray_densenet_model6.h5"
FILE_ID = "1R4tcjg0VC1Oo_rMML7-m_ZG-33KXUnRs"  # Update with your Google Drive file ID
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Check if model exists, if not download it
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("Download complete!")

# Load trained model
model = load_model(MODEL_PATH)

# Image size (must match training size)
IMG_SIZE = 224  

@app.route("/")
def home():
    return render_template("index.html")

# Function to preprocess the image
def preprocess_image(img):
    img = img.convert("RGB")  # Ensure 3 channels
    img = img.resize((IMG_SIZE, IMG_SIZE))  # Resize
    img_array = img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# Prediction function
def predict_xray(img):
    class_labels = ["Benign", "Malignant", "Normal"]
    img_array = preprocess_image(img)
    
    try:
        predictions = model.predict(img_array)  # Get probability scores for all classes
        predicted_class_index = np.argmax(predictions)  # Get highest probability class
        confidence = np.max(predictions)  # Confidence score
        predicted_class = class_labels[predicted_class_index]  # Get class name

        return {"result": predicted_class, "confidence": float(confidence)}
    
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

# Route for image upload & prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    try:
        file = request.files["file"]
        img = Image.open(file)
        prediction = predict_xray(img)  # Get prediction
        return jsonify(prediction)
    
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
