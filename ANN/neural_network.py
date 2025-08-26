"""
Simple NumPy MLP for digit OCR (28x28 -> 10 classes) with backprop.
No external frameworks; good enough for interactive demos.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def xavier_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
	limit = math.sqrt(6.0 / (fan_in + fan_out))
	return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)


@dataclass
class MLPConfig:
	input_size: int = 28 * 28
	hidden_sizes: Tuple[int, ...] = (128,)
	output_size: int = 10
	lr: float = 0.1
	seed: int = 42


class SimpleMLP:
	def __init__(self, cfg: MLPConfig | None = None):
		self.cfg = cfg or MLPConfig()
		self.rng = np.random.default_rng(self.cfg.seed)
		sizes = [self.cfg.input_size, *self.cfg.hidden_sizes, self.cfg.output_size]
		# Weights and biases
		self.W: List[np.ndarray] = []
		self.b: List[np.ndarray] = []
		for i in range(len(sizes) - 1):
			self.W.append(xavier_init(sizes[i], sizes[i + 1], self.rng))
			self.b.append(np.zeros((1, sizes[i + 1]), dtype=np.float32))

	# --- activations ---
	@staticmethod
	def relu(x: np.ndarray) -> np.ndarray:
		return np.maximum(0.0, x)

	@staticmethod
	def relu_grad(x: np.ndarray) -> np.ndarray:
		return (x > 0).astype(np.float32)

	@staticmethod
	def softmax(z: np.ndarray) -> np.ndarray:
		z = z - z.max(axis=1, keepdims=True)
		e = np.exp(z)
		return e / (e.sum(axis=1, keepdims=True) + 1e-8)

	# --- forward/backward ---
	def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
		"""Return (pre-activations Zs, activations As). A0 is X."""
		As = [X]
		Zs = []
		for i in range(len(self.W)):
			Z = As[-1] @ self.W[i] + self.b[i]
			Zs.append(Z)
			if i < len(self.W) - 1:
				A = self.relu(Z)
			else:
				A = self.softmax(Z)
			As.append(A)
		return Zs, As

	def train_step(self, x: np.ndarray, label: int) -> float:
		"""One SGD step on a single example. x shape (784,), label in [0..9]."""
		X = x.reshape(1, -1).astype(np.float32)
		y = np.zeros((1, self.cfg.output_size), dtype=np.float32)
		y[0, int(label)] = 1.0

		Zs, As = self.forward(X)
		y_hat = As[-1]
		# Cross-entropy loss
		loss = -np.sum(y * np.log(y_hat + 1e-8))

		# Backprop
		dA = (y_hat - y)  # softmax + CE gradient
		for i in reversed(range(len(self.W))):
			A_prev = As[i]
			Z = Zs[i]
			dZ = dA
			if i < len(self.W) - 1:  # hidden layer: apply relu grad
				dZ = dA * self.relu_grad(Z)
			dW = A_prev.T @ dZ
			db = dZ.sum(axis=0, keepdims=True)
			dA = dZ @ self.W[i].T

			# Update
			self.W[i] -= self.cfg.lr * dW
			self.b[i] -= self.cfg.lr * db

		return float(loss)

	def predict_proba(self, x: np.ndarray) -> np.ndarray:
		X = x.reshape(1, -1).astype(np.float32)
		_, As = self.forward(X)
		return As[-1][0]  # shape (10,)

	def predict_digit(self, x: np.ndarray) -> Tuple[int, float]:
		p = self.predict_proba(x)
		idx = int(np.argmax(p))
		return idx, float(p[idx])

	def save(self, path: str) -> None:
		np.savez(path, **{f"W{i}": W for i, W in enumerate(self.W)}, **{f"b{i}": b for i, b in enumerate(self.b)})

	def load(self, path: str) -> None:
		data = np.load(path)
		for i in range(len(self.W)):
			self.W[i] = data[f"W{i}"]
			self.b[i] = data[f"b{i}"]

