# OCR

Interactive handwritten digit OCR. Draw on a full‑screen canvas and send your sketch for prediction or as a labeled training sample.

Live demo (GitHub Pages):
- https://jsprouse0.github.io/OCR/

What it is:
- A simple browser app (HTML/CSS/JS) that downsamples drawings to 28×28 grayscale.
- A lightweight Python server (Flask) with JSON endpoints for predict/train.

Status:
- UI is live; server currently returns a placeholder prediction. ANN integration is planned next.
