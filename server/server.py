from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Ensure project root is on sys.path so we can import ANN
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
	sys.path.insert(0, ROOT_DIR)

# Project base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UI_DIR = os.path.join(BASE_DIR, "user interface")
CLIENT_DIR = os.path.join(BASE_DIR, "client")
UI_STATIC_DIR = os.path.join(BASE_DIR, "user interface")
MODEL_PATH = os.path.join(BASE_DIR, "model.npz")

# ANN
import numpy as np
from ANN.ocr import OCRModel, OCRConfig


app = Flask(__name__)
CORS(app)


@dataclass
class Sample:
	image: List[float]  # 28x28 flattened, values 0..1
	label: int          # 0..9


# In-memory store for quick start; replace with ANN training later
TRAINING_DATA: List[Sample] = []
MODEL = OCRModel(OCRConfig(hidden_sizes=(128,)))
TRAIN_COUNT = 0
SAVE_EVERY = 20  # save model after this many online train steps

# Try loading a previously saved model
try:
	if os.path.exists(MODEL_PATH):
		MODEL.load(MODEL_PATH)
except Exception as e:
	print(f"[warn] Failed to load model from {MODEL_PATH}: {e}")


def ann_predict(image: List[float]) -> Tuple[int, float]:
	digit, conf = MODEL.predict_digit(image)
	return digit, conf


# BASE_DIR and other paths defined above


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

	digit, conf = ann_predict(image)
	# Also return full probability distribution for better UI/debug
	probs = MODEL.predict_proba(image).tolist()
	return jsonify({"digit": int(digit), "confidence": float(conf), "probs": probs})


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

	global TRAIN_COUNT
	TRAINING_DATA.append(Sample(image=image, label=label))
	# Perform a small training step for online learning
	loss = MODEL.train_step(image, int(label))
	TRAIN_COUNT += 1
	# periodic save
	if TRAIN_COUNT % SAVE_EVERY == 0:
		try:
			MODEL.save(MODEL_PATH)
		except Exception as e:
			print(f"[warn] Failed to save model: {e}")
	return jsonify({"message": "Sample stored & model updated", "count": len(TRAINING_DATA), "loss": loss})


@app.post("/api/train_all")
def train_all():
	data = request.get_json(silent=True) or {}
	epochs = int(data.get("epochs", 5))
	if not TRAINING_DATA:
		return jsonify({"error": "no samples to train"}), 400
	# Build arrays once
	X = np.array([s.image for s in TRAINING_DATA], dtype=np.float32)
	y = np.array([s.label for s in TRAINING_DATA], dtype=np.int64)
	N = X.shape[0]
	rng = np.random.default_rng(42)
	losses = []
	for _ in range(max(1, epochs)):
		perm = rng.permutation(N)
		epoch_loss = 0.0
		for idx in perm:
			loss = MODEL.train_step(X[idx], int(y[idx]))
			epoch_loss += loss
		losses.append(epoch_loss / N)
	# Save after train_all
	try:
		MODEL.save(MODEL_PATH)
	except Exception as e:
		print(f"[warn] Failed to save model: {e}")
	return jsonify({"message": "Trained over stored samples", "epochs": epochs, "avg_epoch_loss": float(np.mean(losses)), "samples": int(N)})


@app.post("/api/save")
def save_model():
	try:
		MODEL.save(MODEL_PATH)
		return jsonify({"message": "Model saved", "path": MODEL_PATH})
	except Exception as e:
		return jsonify({"error": str(e)}), 500


def main():
	port = int(os.environ.get("PORT", "5000"))
	app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
	main()

