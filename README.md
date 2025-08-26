# OCR

A minimal, interactive OCR playground for handwritten digits. Draw directly in the browser on a full-viewport canvas, then send your sketch to a lightweight Python server for prediction or to store labeled training samples.

## What’s included
- Full-screen drawing UI (HTML/CSS/JS) with grid guides and a compact top-right dropdown menu.
- Client script that downsamples drawings to 28×28 grayscale arrays for ANN-ready input.
- Python Flask server with CORS and JSON endpoints:
	- `POST /api/predict` — returns a placeholder digit + confidence.
	- `POST /api/train` — stores labeled samples in memory.
	- `GET /health` — simple health/status.

The neural network integration will be added next; the current server uses a dummy predictor so you can iterate on the UI and data flow quickly.

## Quick start
1) Create/activate a virtual environment and install deps:
```bash
cd /home/developer/OCR
source .venv/bin/activate  # or: python3 -m venv .venv && source .venv/bin/activate
pip install -U pip Flask flask-cors
```
2) Run the server:
```bash
python server/server.py
```
3) Open the UI:
- Visit http://127.0.0.1:5000

## How it works (high level)
- You draw on a high‑DPI canvas; the client resizes and converts it to a 28×28 normalized array [0..1].
- For prediction: the array is POSTed to the server, which returns a digit guess and confidence (placeholder for now).
- For training: the array plus a label (0–9) is stored server‑side to be used by the upcoming ANN.

## Project structure
```
client/            # Browser client logic (drawing, downsampling, API calls)
server/            # Flask API server and static file routing
user interface/    # HTML + CSS for the UI
ANN/               # ANN code (placeholder, to be wired in next)
```

## Next steps
- Replace the dummy predictor with the actual ANN and wire `/api/train` to update weights.
- Persist training samples and model state to disk.
- Add tests for serialization, inference, and endpoint contracts.
