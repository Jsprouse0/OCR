// Simple OCR client: full-screen canvas, send to server for train/predict

// API base URL; override on static hosting (e.g., GitHub Pages) by setting window.OCR_API_URL
const SERVER_URL = ((window.OCR_API_URL || window.location.origin) + '').replace(/\/$/, '');

// Canvas setup
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");
const trainBtn = document.getElementById("trainBtn");
const trainAllBtn = document.getElementById("trainAllBtn");
const labelInput = document.getElementById("labelInput");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const brandToggle = document.getElementById("brandToggle");
const dropdown = document.getElementById("dropdown");

// Drawing config
// Canvas and drawing settings
const DOWNSCALE = 28; // target 28x28 for OCR input
const BRUSH_SIZE = 18; // base brush width in CSS pixels; scaled by devicePixelRatio
let drawing = false;
let dpr = Math.max(1, window.devicePixelRatio || 1);

function resizeCanvas() {
	// Size backing store to viewport in device pixels for crisp lines
	const cssWidth = window.innerWidth;
	const cssHeight = window.innerHeight;
	dpr = Math.max(1, window.devicePixelRatio || 1);
	canvas.width = Math.floor(cssWidth * dpr);
	canvas.height = Math.floor(cssHeight * dpr);
	// Clear background to white and redraw grid
	ctx.setTransform(1, 0, 0, 1, 0, 0);
	ctx.fillStyle = "white";
	ctx.fillRect(0, 0, canvas.width, canvas.height);
	drawGrid();
	// Stroke style
	ctx.lineCap = "round";
	ctx.lineJoin = "round";
	ctx.strokeStyle = "black";
	ctx.lineWidth = BRUSH_SIZE * dpr;
}

function initCanvas() {
	resizeCanvas();
}

function getPos(evt) {
	const rect = canvas.getBoundingClientRect();
	const clientX = (evt.touches ? evt.touches[0].clientX : evt.clientX);
	const clientY = (evt.touches ? evt.touches[0].clientY : evt.clientY);
	const scaleX = canvas.width / rect.width;   // account for DPR
	const scaleY = canvas.height / rect.height; // account for DPR
	const x = (clientX - rect.left) * scaleX;
	const y = (clientY - rect.top) * scaleY;
	return { x, y };
}

function startDraw(evt) {
	drawing = true;
	const { x, y } = getPos(evt);
	ctx.beginPath();
	ctx.moveTo(x, y);
	evt.preventDefault();
}

function draw(evt) {
	if (!drawing) return;
	const { x, y } = getPos(evt);
	ctx.lineTo(x, y);
	ctx.stroke();
	evt.preventDefault();
}

function endDraw(evt) {
	drawing = false;
	evt && evt.preventDefault();
}

function clearCanvas() {
	ctx.save();
	ctx.setTransform(1, 0, 0, 1, 0, 0);
	ctx.clearRect(0, 0, canvas.width, canvas.height);
	ctx.restore();
	ctx.fillStyle = "white";
	ctx.fillRect(0, 0, canvas.width, canvas.height);
	drawGrid();
	resultEl.textContent = "";
}

// Convert canvas to 28x28 grayscale array [0..1]
function getImageArray() {
	const off = document.createElement("canvas");
	off.width = DOWNSCALE;
	off.height = DOWNSCALE;
	const octx = off.getContext("2d");
	// downscale with built-in drawImage; ensure white background
	octx.fillStyle = "white";
	octx.fillRect(0, 0, off.width, off.height);
	octx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, off.width, off.height);
	const img = octx.getImageData(0, 0, off.width, off.height);
	const data = img.data;
	const arr = new Array(DOWNSCALE * DOWNSCALE);
	for (let i = 0, j = 0; i < data.length; i += 4, j++) {
		const r = data[i], g = data[i + 1], b = data[i + 2];
		// Convert to grayscale (luminosity)
		const gray = 0.299 * r + 0.587 * g + 0.114 * b; // 0..255
		// Invert to make black strokes -> 1.0
		const v = 1 - gray / 255;
		arr[j] = +v.toFixed(4);
	}
	return arr;
}

// Draw a subtle grid to guide drawing; major grid every 8 cells
function drawGrid() {
	const w = canvas.width, h = canvas.height;
	const g = 28; // logical target cells
	const stepX = w / g;
	const stepY = h / g;
	ctx.save();
	ctx.lineWidth = 1 * dpr;
	// minor lines
	ctx.strokeStyle = "rgba(0,0,0,0.08)";
	ctx.beginPath();
	for (let i = 1; i < g; i++) {
		const x = Math.round(i * stepX) + 0.5; // crisp
		ctx.moveTo(x, 0);
		ctx.lineTo(x, h);
	}
	for (let j = 1; j < g; j++) {
		const y = Math.round(j * stepY) + 0.5;
		ctx.moveTo(0, y);
		ctx.lineTo(w, y);
	}
	ctx.stroke();
	// major lines every 7 cells
	ctx.strokeStyle = "rgba(0,0,0,0.16)";
	ctx.beginPath();
	const major = 7;
	for (let i = major; i < g; i += major) {
		const x = Math.round(i * stepX) + 0.5;
		ctx.moveTo(x, 0);
		ctx.lineTo(x, h);
	}
	for (let j = major; j < g; j += major) {
		const y = Math.round(j * stepY) + 0.5;
		ctx.moveTo(0, y);
		ctx.lineTo(w, y);
	}
	ctx.stroke();
	ctx.restore();
}

async function predict() {
	const image = getImageArray();
	setStatus("Predicting…");
	try {
		const res = await fetch(`${SERVER_URL}/api/predict`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ image })
		});
		if (!res.ok) throw new Error(`HTTP ${res.status}`);
		const data = await res.json();
		if (Array.isArray(data.probs)) {
			// Sort top-3 predictions
			const entries = data.probs.map((p, i) => ({ i, p }));
			entries.sort((a, b) => b.p - a.p);
			const top = entries.slice(0, 3).map(e => `${e.i}:${(e.p*100).toFixed(1)}%`).join("  ");
			resultEl.textContent = `Prediction: ${data.digit}  (${(data.confidence*100).toFixed(1)}%)  |  top-3 ${top}`;
		} else {
			resultEl.textContent = `Prediction: ${data.digit}  (confidence: ${(data.confidence*100).toFixed(1)}%)`;
		}
	} catch (err) {
		resultEl.textContent = `Error: ${err.message}`;
	} finally {
		setStatus("");
	}
}

async function train() {
	const label = Number(labelInput.value);
	if (!Number.isInteger(label) || label < 0 || label > 9) {
		resultEl.textContent = "Enter a label 0-9 to train.";
		return;
	}
	const image = getImageArray();
	setStatus("Sending training sample…");
	try {
		const res = await fetch(`${SERVER_URL}/api/train`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ image, label })
		});
		if (!res.ok) throw new Error(`HTTP ${res.status}`);
		const data = await res.json();
		const extra = (typeof data.loss === 'number') ? `  loss=${data.loss.toFixed(3)}` : '';
		const count = (typeof data.count === 'number') ? `  samples=${data.count}` : '';
		resultEl.textContent = (data.message || "Training sample stored.") + extra + count;
	} catch (err) {
		resultEl.textContent = `Error: ${err.message}`;
	} finally {
		setStatus("");
	}
}

async function trainAll() {
	setStatus("Training over stored samples…");
	try {
		const res = await fetch(`${SERVER_URL}/api/train_all`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ epochs: 10 })
		});
		if (!res.ok) throw new Error(`HTTP ${res.status}`);
		const data = await res.json();
		resultEl.textContent = `${data.message}  epochs=${data.epochs}  avgLoss=${Number(data.avg_epoch_loss).toFixed(3)}  samples=${data.samples}`;
	} catch (err) {
		resultEl.textContent = `Error: ${err.message}`;
	} finally {
		setStatus("");
	}
}

function setStatus(msg) {
	statusEl.textContent = msg;
}

function bindEvents() {
	// Mouse
	canvas.addEventListener("mousedown", startDraw);
	canvas.addEventListener("mousemove", draw);
	window.addEventListener("mouseup", endDraw);
	// Touch
	canvas.addEventListener("touchstart", startDraw, { passive: false });
	canvas.addEventListener("touchmove", draw, { passive: false });
	window.addEventListener("touchend", endDraw, { passive: false });

	clearBtn.addEventListener("click", clearCanvas);
	predictBtn.addEventListener("click", predict);
	trainBtn.addEventListener("click", train);
	trainAllBtn?.addEventListener("click", trainAll);
	window.addEventListener("resize", () => {
		resizeCanvas();
	});

		// Prevent accidental scroll/arrow increment on number input
		labelInput.addEventListener("wheel", (e) => e.preventDefault(), { passive: false });
		labelInput.addEventListener("keydown", (e) => {
			const blocked = ["ArrowUp", "ArrowDown", "PageUp", "PageDown"]; 
			if (blocked.includes(e.key) && !e.ctrlKey && !e.metaKey) e.preventDefault();
		});

		// Dropdown behavior
		brandToggle?.addEventListener("click", () => {
			const isOpen = dropdown.hasAttribute("hidden") ? false : true;
			if (isOpen) {
				dropdown.setAttribute("hidden", "");
				brandToggle.setAttribute("aria-expanded", "false");
			} else {
				dropdown.removeAttribute("hidden");
				brandToggle.setAttribute("aria-expanded", "true");
			}
		});
		document.addEventListener("click", (e) => {
			if (!dropdown) return;
			const btn = brandToggle;
			const tgt = e.target;
			if (!btn.contains(tgt) && !dropdown.contains(tgt)) {
				if (!dropdown.hasAttribute("hidden")) {
					dropdown.setAttribute("hidden", "");
					brandToggle.setAttribute("aria-expanded", "false");
				}
			}
		});
}

// Init
initCanvas();
bindEvents();

