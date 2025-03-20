import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_file, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from PIL import Image

# ✅ Initialize Flask App
app = Flask(__name__, template_folder="templates")

# ✅ Define Model Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "brain_tumor_model.keras")

# ✅ Load Trained Model
try:
    model = load_model(MODEL_PATH)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error Loading Model: {e}")
    model = None

# ✅ Define Class Labels
CLASS_LABELS = ["Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor", "No Tumor"]

# ✅ Function to Estimate Tumor Severity
def estimate_tumor_severity(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    tumor_area = np.sum(binary == 255)

    if tumor_area < 5000:
        return "Small Tumor"
    elif tumor_area < 15000:
        return "Medium Tumor"
    else:
        return "Large Tumor"

# ✅ Function to Generate Grad-CAM Heatmap
def generate_grad_cam(img_path):
    img = Image.open(img_path).convert("RGB").resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    grad_model = Model(inputs=[model.inputs], outputs=[model.get_layer("block5_conv3").output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions)]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs.numpy()[0]
    heatmap = np.mean(conv_outputs * pooled_grads.numpy(), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (128, 128))
    heatmap = np.uint8(255 * heatmap)

    heatmap_path = f"uploads/heatmap_{os.path.basename(img_path)}"
    cv2.imwrite(heatmap_path, heatmap)
    
    return heatmap_path

# ✅ Function to Process and Predict Image
def process_and_predict(img_path):
    if model is None:
        return "Error: Model not loaded", None, None, None

    img = Image.open(img_path).convert("RGB").resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = CLASS_LABELS[np.argmax(predictions)]
    confidence_score = round(np.max(predictions) * 100, 2)

    severity = estimate_tumor_severity(img_path)
    heatmap_path = generate_grad_cam(img_path)

    return predicted_class, confidence_score, severity, heatmap_path

# ✅ Route for Upload Page
@app.route("/")
def upload_page():
    return render_template("upload.html")

# ✅ Route for Handling Image Upload and Prediction
@app.route("/predict", methods=["POST"])
def predict_tumor():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    file.save(img_path)

    predicted_class, confidence_score, severity, heatmap_path = process_and_predict(img_path)

    return render_template(
        "result.html",
        filename=file.filename,
        predicted_class=predicted_class,
        confidence_score=confidence_score,
        severity=severity,
        heatmap_url=f"/gradcam/{os.path.basename(heatmap_path)}"
    )

# ✅ Route for Serving Grad-CAM Image
@app.route("/gradcam/<filename>")
def get_gradcam_image(filename):
    image_path = os.path.join("uploads", filename)
    if os.path.exists(image_path):
        return send_file(image_path, mimetype="image/png")
    else:
        return jsonify({"error": "Grad-CAM image not found"}), 404

# ✅ Run Flask Server
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
