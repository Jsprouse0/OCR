from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


@dataclass
class Sample:
	image: List[float]  # 28x28 flattened, values 0..1
	label: int          # 0..9


# In-memory store for quick start; replace with ANN training later
TRAINING_DATA: List[Sample] = []


def dummy_predict(image: List[float]) -> Tuple[int, float]:
	"""A placeholder prediction function.
	Returns a fake digit and confidence until ANN is implemented.
	"""
	# Heuristic: use average intensity to flip between two digits
	avg = sum(image) / max(1, len(image))
	digit = 8 if avg > 0.2 else 1
	# Confidence based on how far from midpoint
	conf = min(0.99, max(0.51, abs(avg - 0.2) + 0.5))
	return digit, conf


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UI_DIR = os.path.join(BASE_DIR, "user interface")
CLIENT_DIR = os.path.join(BASE_DIR, "client")
UI_STATIC_DIR = os.path.join(BASE_DIR, "user interface")


@app.get("/health")
def health():
	return jsonify({"status": "ok", "training_samples": len(TRAINING_DATA)})


@app.get("/")
def ui_index():
	# Serve the drawing UI
	return send_from_directory(UI_DIR, "ocr.html")


@app.get("/client/<path:filename>")
def client_assets(filename: str):
	# Serve client-side JS
	return send_from_directory(CLIENT_DIR, filename)


@app.get("/ui/<path:filename>")
def ui_assets(filename: str):
	return send_from_directory(UI_STATIC_DIR, filename)


@app.post("/api/predict")
def predict():
	data = request.get_json(silent=True) or {}
	image = data.get("image")
	if not isinstance(image, list) or len(image) != 28 * 28:
		return jsonify({"error": "image must be a 28x28 flattened list of length 784"}), 400
	try:
		image = [float(v) for v in image]
	except Exception:
		return jsonify({"error": "image values must be numeric"}), 400

	digit, conf = dummy_predict(image)
	return jsonify({"digit": int(digit), "confidence": float(conf)})


@app.post("/api/train")
def train():
	data = request.get_json(silent=True) or {}
	image = data.get("image")
	label = data.get("label")
	if not isinstance(image, list) or len(image) != 28 * 28:
		return jsonify({"error": "image must be a 28x28 flattened list of length 784"}), 400
	if not isinstance(label, int) or not (0 <= label <= 9):
		return jsonify({"error": "label must be an integer 0-9"}), 400
	try:
		image = [float(v) for v in image]
	except Exception:
		return jsonify({"error": "image values must be numeric"}), 400

	TRAINING_DATA.append(Sample(image=image, label=label))
	return jsonify({"message": "Sample stored", "count": len(TRAINING_DATA)})


def main():
	port = int(os.environ.get("PORT", "5000"))
	app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
	main()

