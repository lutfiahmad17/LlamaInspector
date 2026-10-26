import os
import io
import json
import base64
import urllib.request
import urllib.error
import uuid

from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "llava:latest"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/inspect", methods=["POST"])
def inspect_image():
    uploaded = request.files.get("image") or request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"success": False, "message": "Please select an image."}), 400

    filename = secure_filename(uploaded.filename)
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"success": False, "message": "Please upload JPG, PNG, or WEBP."}), 400

    image_bytes = uploaded.read()
    if not image_bytes:
        return jsonify({"success": False, "message": "The image is empty."}), 400

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.format not in ("JPEG", "PNG", "WEBP"):
                raise ValueError("Unsupported image format.")
            img.verify()
    except Exception:
        return jsonify({"success": False, "message": "The image file is invalid."}), 400

    saved_name = str(uuid.uuid4()) + "." + extension
    with open(os.path.join(UPLOAD_FOLDER, saved_name), "wb") as f:
        f.write(image_bytes)

        prompt = (
        "You are Llama Inspector, an AI assistant for visual inspection "
        "of electronic circuit boards. Analyze only what is visible. "
        "Use these five sections in your response:\n\n"
        "1. VISIBLE OBSERVATIONS: Describe visible components, connectors, "
        "pins, labels, and physical features. Read markings only when clear.\n\n"
        "2. POSSIBLE ISSUES: Report visible irregularities only when supported "
        "by evidence. If none are apparent, say no obvious issue was identified.\n\n"
        "3. UNCONFIRMED DETAILS: Explain what cannot be identified reliably. "
        "Never invent chip types, specifications, or component functions.\n\n"
        "4. SAFE NEXT STEPS: Suggest non-destructive checks. Never instruct "
        "the user to handle live circuits or mains voltage.\n\n"
        "5. SAFETY LIMITATION: An image cannot prove electrical safety or "
        "confirm that a board works.\n\n"
        "An ESP32 marking alone does not prove a chip is EEPROM. "
        "Separate observations from hypotheses. If the image is not "
        "electronics, describe what is actually visible instead."
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [base64.b64encode(image_bytes).decode("ascii")],
        "stream": False
    }

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
        analysis = result.get("response", "").strip()
        if not analysis:
            analysis = "The AI returned an empty response. Please try again."
        return jsonify({
            "success": True,
            "analysis": analysis,
            "message": analysis,
            "filename": saved_name,
            "image_url": "/uploads/" + saved_name
        })
    except urllib.error.URLError as error:
        return jsonify({
            "success": False,
            "message": "Could not contact Ollama. Make sure it is running. " + str(error)
        }), 503
    except Exception as error:
        return jsonify({"success": False, "message": "Inspection failed: " + str(error)}), 500


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.errorhandler(413)
def too_large(error):
    return jsonify({"success": False, "message": "Maximum image size is 8 MB."}), 413


if __name__ == "__main__":
    print("Llama Inspector is starting...")
    print("Ollama model:", MODEL_NAME)
    app.run(host="127.0.0.1", port=5000, debug=False)
