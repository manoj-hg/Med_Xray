"""
Flask backend server for Medical AI frontend.
Serves trained model predictions, status, and file downloads.
"""
import os
import sys
import json
import glob
import tempfile
import subprocess
import time
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Path to the trained model directory
CANDIDATES = [
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_training_suite")),
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_training_suite")),
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rocov2-multimodal-training-&-model-artifacts-suite")),
]
TRAINING_SUITE_DIR = next((c for c in CANDIDATES if os.path.exists(c)), CANDIDATES[0])
MODEL_DIR = os.path.join(TRAINING_SUITE_DIR, "trained_model")
PREDICT_SCRIPT = os.path.join(TRAINING_SUITE_DIR, "predict.py")

# Also load CUI mapping for enriched responses
CUI_NAMES = {}
cui_map_path = os.path.join(MODEL_DIR, "cui_mapping.csv")

def load_cui_names():
    global CUI_NAMES
    if os.path.exists(cui_map_path):
        import pandas as pd
        d = pd.read_csv(cui_map_path)
        CUI_NAMES = dict(zip(d.iloc[:, 0].astype(str), d.iloc[:, 1].astype(str)))

def format_bytes(b):
    if b == 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.1f} {unit}" if b >= 10 else f"{b:.2f} {unit}"
    return f"{b:.2f} TB"

def check_medical_image_validity(image_path):
    """
    Guardrail to detect non-medical everyday color photographs (e.g. animals, landscapes, cars).
    Clinical radiology scans (CT, X-Ray, MRI, Angiography, Ultrasound) are natively monochromatic / grayscale
    with very low inter-channel dispersion (color_std <= 8).
    """
    try:
        from PIL import Image
        import numpy as np
        im = Image.open(image_path).convert("RGB")
        arr = np.array(im, dtype=np.float32)
        channel_std = np.std(arr, axis=2)
        mean_std = float(np.mean(channel_std))

        hsv = im.convert("HSV")
        sat = np.array(hsv)[:, :, 1]
        sat_ratio = float(np.mean(sat > 50))

        if mean_std > 12.0 and sat_ratio > 0.18:
            return False, "Non-Medical Image Detected: The uploaded image appears to be an everyday color photograph (e.g. animal, nature, or object). This diagnostic AI is strictly calibrated for radiology modalities (CT, MRI, X-Ray, Ultrasound, Angiogram). Please upload a clinical radiograph."
        return True, ""
    except Exception:
        return True, ""



@app.route("/api/status", methods=["GET"])
def status():
    """Return training status, model comparison, histories, configs, etc."""
    try:
        files = []
        if os.path.exists(MODEL_DIR):
            for entry in os.listdir(MODEL_DIR):
                full_path = os.path.join(MODEL_DIR, entry)
                if os.path.isfile(full_path):
                    size = os.path.getsize(full_path)
                    category = "metrics"
                    if entry.endswith(".pt") or entry.endswith(".zip"):
                        category = "weights"
                    elif entry.endswith("_config.json"):
                        category = "config"
                    elif entry in ("label_vocab.json", "tfidf.pkl", "cui_mapping.csv"):
                        category = "vocab"
                    elif entry.endswith(".py") or entry == "README.md":
                        category = "script"
                    files.append({
                        "name": entry,
                        "sizeBytes": size,
                        "sizeFormatted": format_bytes(size),
                        "category": category,
                        "url": f"/api/download/{entry}",
                    })

        # Parse comparison.csv
        comparison = []
        cmp_path = os.path.join(MODEL_DIR, "comparison.csv")
        if os.path.exists(cmp_path):
            import pandas as pd
            df = pd.read_csv(cmp_path)
            comparison = df.to_dict(orient="records")

        # Parse histories
        histories = {}
        for m in ["cnn", "resnet50", "vit", "fusion"]:
            h_path = os.path.join(MODEL_DIR, f"{m}_history.csv")
            if os.path.exists(h_path):
                import pandas as pd
                histories[m] = pd.read_csv(h_path).to_dict(orient="records")

        # Parse configs
        configs = {}
        for m in ["cnn", "resnet50", "vit", "fusion"]:
            c_path = os.path.join(MODEL_DIR, f"{m}_config.json")
            if os.path.exists(c_path):
                with open(c_path) as f:
                    configs[m] = json.load(f)

        # Confusion matrices & sample cases
        cm_path = os.path.join(MODEL_DIR, "confusion_matrices.json")
        confusion_matrices = json.load(open(cm_path)) if os.path.exists(cm_path) else []

        sc_path = os.path.join(MODEL_DIR, "sample_cases.json")
        sample_cases = json.load(open(sc_path)) if os.path.exists(sc_path) else []

        return jsonify({
            "ready": len(comparison) >= 4,
            "files": files,
            "comparison": comparison,
            "histories": histories,
            "configs": configs,
            "confusionMatrices": confusion_matrices,
            "sampleCases": sample_cases,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/download/<filename>", methods=["GET"])
def download(filename):
    """Download a model artifact file."""
    file_path = os.path.join(MODEL_DIR, os.path.basename(filename))
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, as_attachment=True)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Run inference on an uploaded image.
    Accepts: multipart/form-data with 'image' file, optional 'model' and 'caption' fields.
    Also accepts JSON with 'imageBase64', 'model', 'caption', 'sampleId'.
    """
    load_cui_names()

    selected_model = "fusion"
    caption = ""
    topn = 10
    temp_file = None
    image_path = None

    allowed_models = ["cnn", "resnet50", "vit", "fusion"]

    if request.content_type and "multipart/form-data" in request.content_type:
        # Form upload
        selected_model = request.form.get("model", "fusion")
        caption = request.form.get("caption", "")
        topn = int(request.form.get("topn", 10))

        if "image" in request.files:
            f = request.files["image"]
            temp_file = os.path.join(tempfile.gettempdir(), f"upload_{int(time.time()*1000)}.jpg")
            f.save(temp_file)
            image_path = temp_file
    else:
        # JSON body
        data = request.get_json(silent=True) or {}
        selected_model = data.get("model", "fusion")
        caption = data.get("caption", "")
        topn = data.get("topn", 10)

        if data.get("imageBase64"):
            import base64
            b64 = data["imageBase64"]
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            temp_file = os.path.join(tempfile.gettempdir(), f"upload_{int(time.time()*1000)}.jpg")
            with open(temp_file, "wb") as f:
                f.write(base64.b64decode(b64))
            image_path = temp_file
        elif data.get("sampleId"):
            candidate = os.path.join(MODEL_DIR, "samples", f"{data['sampleId']}.jpg")
            if os.path.exists(candidate):
                image_path = candidate

    if selected_model not in allowed_models:
        selected_model = "fusion"

    # Fallback to first sample image
    if not image_path or not os.path.exists(image_path):
        samples_dir = os.path.join(MODEL_DIR, "samples")
        if os.path.exists(samples_dir):
            imgs = [f for f in os.listdir(samples_dir) if f.endswith(".jpg")]
            if imgs:
                image_path = os.path.join(samples_dir, imgs[0])

    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": "No valid image found for inference"}), 400

    # Guardrail: Check for Out-of-Distribution (OOD) non-medical everyday color photos
    if temp_file and os.path.exists(temp_file):
        is_medical, rejection_reason = check_medical_image_validity(image_path)
        if not is_medical:
            try:
                os.unlink(temp_file)
            except Exception:
                pass
            return jsonify({
                "error": rejection_reason,
                "isNonMedical": True
            }), 400

    # Run predict.py
    args = [
        sys.executable, PREDICT_SCRIPT,
        "--model_dir", MODEL_DIR,
        "--model", selected_model,
        "--image", image_path,
        "--caption", caption or "",
        "--topn", str(topn),
    ]

    t0 = time.time()
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=TRAINING_SUITE_DIR,
        )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Prediction timed out"}), 500
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except:
                pass

    if result.returncode != 0:
        return jsonify({"error": result.stderr or "Prediction failed"}), 500

    # Parse output lines: "  0.123  C1234567  Some Name"
    import re
    predictions = []
    for line in result.stdout.strip().split("\n"):
        m = re.match(r"^\s*([0-9.]+)\s+(C\d+)\s*(.*)$", line)
        if m:
            cui_code = m.group(2)
            predictions.append({
                "probability": float(m.group(1)),
                "cui": cui_code,
                "name": m.group(3).strip() or CUI_NAMES.get(cui_code, cui_code),
            })

    latency = int((time.time() - t0) * 1000)

    return jsonify({
        "model": selected_model,
        "latencyMs": latency,
        "predictions": predictions,
        "rawOutput": result.stdout,
    })


@app.route("/api/sample-image/<sample_id>", methods=["GET"])
def sample_image(sample_id):
    """Serve a sample image."""
    file_path = os.path.join(MODEL_DIR, "samples", f"{os.path.basename(sample_id)}.jpg")
    if not os.path.exists(file_path):
        return jsonify({"error": "Sample not found"}), 404
    return send_file(file_path, mimetype="image/jpeg")


if __name__ == "__main__":
    load_cui_names()
    print(f"Model directory: {MODEL_DIR}")
    print(f"Predict script: {PREDICT_SCRIPT}")
    print(f"Models ready: {os.path.exists(os.path.join(MODEL_DIR, 'comparison.csv'))}")
    app.run(host="0.0.0.0", port=5000, debug=True)
