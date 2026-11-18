"""Pure Python Multi-Layer Perceptron (MLP) for Squirrel Policy.

Implements a lightweight feed-forward neural network with parameter flattening,
serialization, and fast forward inference without external dependencies.
"""
import math
import random
import json
from typing import List, Tuple, Dict, Any


def tanh(x: float) -> float:
    """Clamped tanh activation to avoid overflow."""
    if x > 20.0:
        return 1.0
    elif x < -20.0:
        return -1.0
    return math.tanh(x)


class PolicyMLP:
    """Compact policy network mapping observations to movement actions."""

    def __init__(self, layer_sizes: Tuple[int, ...] = (12, 20, 20, 5), weights: List[float] = None):
        self.layer_sizes = layer_sizes
        self.num_weights = self._calculate_num_weights(layer_sizes)

        if weights is not None:
            if len(weights) != self.num_weights:
                raise ValueError(f"Expected {self.num_weights} weights, got {len(weights)}")
            self.weights = list(weights)
        else:
            # Initialize with small random weights (Xavier-like uniform)
            self.weights = self._init_random_weights()

        # Unpack weights into layer matrices and bias vectors
        self.W: List[List[List[float]]] = []
        self.B: List[List[float]] = []
        self._unpack_weights()

    @staticmethod
    def _calculate_num_weights(layer_sizes: Tuple[int, ...]) -> int:
        total = 0
        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            total += (n_in * n_out) + n_out  # Weights + Biases
        return total

    def _init_random_weights(self) -> List[float]:
        weights = []
        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            scale = math.sqrt(2.0 / (n_in + n_out))
            for _ in range(n_in * n_out + n_out):
                weights.append(random.gauss(0.0, scale))
        return weights

    def _unpack_weights(self):
        """Unpack 1D weight list into 2D layer matrices and bias vectors."""
        self.W.clear()
        self.B.clear()
        offset = 0
        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]

            # Weights: matrix of shape (n_out, n_in)
            layer_w = []
            for _ in range(n_out):
                row = self.weights[offset : offset + n_in]
                layer_w.append(row)
                offset += n_in
            self.W.append(layer_w)

            # Biases: vector of shape (n_out,)
            layer_b = self.weights[offset : offset + n_out]
            offset += n_out
            self.B.append(layer_b)

    def set_weights(self, weights: List[float]):
        """Update flattened weights and rebuild matrices."""
        if len(weights) != self.num_weights:
            raise ValueError(f"Expected {self.num_weights} weights, got {len(weights)}")
        self.weights = list(weights)
        self._unpack_weights()

    def forward(self, x: List[float]) -> List[float]:
        """Compute forward pass through network. Returns raw output logits."""
        curr = x
        num_layers = len(self.W)

        for l in range(num_layers):
            weights = self.W[l]
            biases = self.B[l]
            is_last = (l == num_layers - 1)

            next_curr = []
            for j in range(len(biases)):
                # Dot product between row j and input vector curr
                w_row = weights[j]
                val = biases[j]
                for k in range(len(curr)):
                    val += w_row[k] * curr[k]

                if not is_last:
                    val = tanh(val)
                next_curr.append(val)

            curr = next_curr

        return curr

    def predict(self, x: List[float], deterministic: bool = True) -> int:
        """Select action with highest logit (or sample from softmax)."""
        logits = self.forward(x)
        if deterministic:
            # Argmax
            best_idx = 0
            best_val = logits[0]
            for i in range(1, len(logits)):
                if logits[i] > best_val:
                    best_val = logits[i]
                    best_idx = i
            return best_idx
        else:
            # Softmax sampling
            max_l = max(logits)
            exp_l = [math.exp(l - max_l) for l in logits]
            sum_exp = sum(exp_l)
            probs = [e / sum_exp for e in exp_l]
            r = random.random()
            cum = 0.0
            for i, p in enumerate(probs):
                cum += p
                if r <= cum:
                    return i
            return len(logits) - 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model for persistence."""
        return {
            "layer_sizes": list(self.layer_sizes),
            "weights": self.weights,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyMLP":
        """Deserialize model from dictionary."""
        return cls(
            layer_sizes=tuple(data["layer_sizes"]),
            weights=data["weights"],
        )

    def save_json(self, filepath: str):
        """Save model to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> "PolicyMLP":
        """Load model from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
