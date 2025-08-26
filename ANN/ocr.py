from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from .neural_network import SimpleMLP, MLPConfig


def _to_vec(image: Iterable[float]) -> np.ndarray:
	arr = np.asarray(list(image), dtype=np.float32)
	if arr.size != 28 * 28:
		raise ValueError("image must have 784 elements (28x28 flattened)")
	# normalize bounds [0,1]
	arr = np.clip(arr, 0.0, 1.0)
	return arr


@dataclass
class OCRConfig:
	hidden_sizes: Tuple[int, ...] = (128,)
	lr: float = 0.1
	seed: int = 42


class OCRModel:
	def __init__(self, cfg: OCRConfig | None = None):
		cfg = cfg or OCRConfig()
		self.net = SimpleMLP(MLPConfig(hidden_sizes=cfg.hidden_sizes, lr=cfg.lr, seed=cfg.seed))

	# Training
	def train_step(self, image: Iterable[float], label: int) -> float:
		x = _to_vec(image)
		return self.net.train_step(x, int(label))

	# Inference
	def predict_digit(self, image: Iterable[float]) -> Tuple[int, float]:
		x = _to_vec(image)
		return self.net.predict_digit(x)

	def predict_proba(self, image: Iterable[float]) -> np.ndarray:
		x = _to_vec(image)
		return self.net.predict_proba(x)

	# Persistence
	def save(self, path: str) -> None:
		self.net.save(path)

	def load(self, path: str) -> None:
		self.net.load(path)


__all__ = ["OCRModel", "OCRConfig"]

