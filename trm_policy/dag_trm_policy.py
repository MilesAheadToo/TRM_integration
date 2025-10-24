"""
Adapters that allow the Tiny Recursive Model (TRM) to act as an order policy
inside The Beer Game DAG simulation.

The module intentionally keeps the public surface small so it can be imported
directly from the main Beer Game repository::

    from trm_policy.dag_trm_policy import TRMSupplyPolicy

The policy consumes the same observation dictionaries that the in-repo policies
receive (see :mod:`backend.app.services.policies`).  Observations are encoded
into a fixed-length integer sequence, passed through a TRM instance, and the
decoded prediction is turned into a supply order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence

from collections import deque

try:
    import torch
    from torch import Tensor
except Exception as exc:  # pragma: no cover - torch is an optional runtime dep
    raise RuntimeError(
        "The TRM integration package requires PyTorch. "
        "Install it with `pip install torch` before using TRMSupplyPolicy."
    ) from exc

try:
    from tiny_recursive_model import MLPMixer1D, TinyRecursiveModel
except Exception as exc:  # pragma: no cover - the submodule might be missing
    raise RuntimeError(
        "The Tiny Recursive Model dependency is missing. "
        "Add the lucidrains/tiny-recursive-model submodule or install the "
        "package so that `from tiny_recursive_model import TinyRecursiveModel` works."
    ) from exc


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    """Describe how to encode a scalar observation into a token.

    Attributes:
        name: Name of the observation key (e.g. ``inventory``).
        minimum: Lower bound used for clipping before quantisation.
        maximum: Upper bound used for clipping before quantisation.
        missing_token: Token to emit when the observation is not present.  When
            ``None`` the mid-point token between ``minimum`` and ``maximum`` is
            used.
    """

    name: str
    minimum: float
    maximum: float
    missing_token: Optional[int] = None

    def encode(self, value: Any, *, num_tokens: int) -> int:
        """Quantise ``value`` into the ``[0, num_tokens - 1]`` token range."""

        if value is None:
            return self._missing_token(num_tokens)

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return self._missing_token(num_tokens)

        clipped = min(max(numeric, self.minimum), self.maximum)
        if self.maximum == self.minimum:
            return 0

        scale = (clipped - self.minimum) / (self.maximum - self.minimum)
        return int(round(scale * (num_tokens - 1)))

    def decode(self, token: int, *, num_tokens: int) -> float:
        """Map a token back to a floating point approximation."""

        token = max(0, min(int(token), num_tokens - 1))
        scale = token / (num_tokens - 1)
        return self.minimum + scale * (self.maximum - self.minimum)

    def _missing_token(self, num_tokens: int) -> int:
        if self.missing_token is not None:
            return int(self.missing_token)
        return num_tokens // 2


class ObservationEncoder:
    """Encode Beer Game observations into TRM-friendly integer sequences."""

    def __init__(
        self,
        features: Sequence[FeatureSpec],
        *,
        num_tokens: int = 256,
        pad_token: int | None = None,
    ) -> None:
        if not features:
            raise ValueError("At least one feature specification is required.")
        self.features = list(features)
        self.num_tokens = int(num_tokens)
        self.pad_token = int(pad_token) if pad_token is not None else 0

    @property
    def feature_count(self) -> int:
        return len(self.features)

    def encode_observation(self, obs: Dict[str, Any]) -> List[int]:
        """Encode a single observation dictionary into tokens."""

        return [
            spec.encode(obs.get(spec.name), num_tokens=self.num_tokens)
            for spec in self.features
        ]

    def decode_feature(self, index: int, token: int) -> float:
        spec = self.features[index]
        return spec.decode(token, num_tokens=self.num_tokens)


class TRMHistoryBuffer:
    """Maintain a bounded history of encoded observations."""

    def __init__(self, encoder: ObservationEncoder, *, max_history: int) -> None:
        if max_history <= 0:
            raise ValueError("max_history must be a positive integer.")
        self.encoder = encoder
        self.max_history = int(max_history)
        self._history: Deque[List[int]] = deque(maxlen=self.max_history)

    def append(self, obs: Dict[str, Any]) -> None:
        self._history.append(self.encoder.encode_observation(obs))

    def clear(self) -> None:
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)

    def tokens(self) -> List[int]:
        """Return the flattened history with oldest observations first."""

        flattened: List[int] = []
        for encoded_obs in self._history:
            flattened.extend(encoded_obs)
        return flattened


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def build_default_trm_model(
    *,
    dim: int = 32,
    depth: int = 4,
    seq_len: int,
    num_tokens: int = 256,
) -> TinyRecursiveModel:
    """Construct a lightweight TRM suitable for Beer Game sequences."""

    network = MLPMixer1D(dim=dim, depth=depth, seq_len=seq_len)
    return TinyRecursiveModel(dim=dim, num_tokens=num_tokens, network=network)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class TRMSupplyPolicy:
    """Stateful supply policy that defers the decision to a TRM instance.

    Parameters:
        model: Trained :class:`TinyRecursiveModel` ready for inference.  When
            ``None`` a randomly initialised model is created via
            :func:`build_default_trm_model`.
        feature_specs: Description of which observation fields are encoded.
        history_length: Number of timesteps to retain before invoking TRM.
        min_history: Require this many historical points before trusting TRM.
        device: Torch device string for inference.
        seq_len: Sequence length expected by the TRM.  Must be at least
            ``history_length * len(feature_specs)``.
        checkpoint_path: Optional path to a ``state_dict`` to load into the
            model.
        order_floor: Hard lower bound on the returned order quantity.
    """

    DEFAULT_FEATURES: Sequence[FeatureSpec] = (
        FeatureSpec("inventory", -100.0, 200.0),
        FeatureSpec("backlog", 0.0, 200.0),
        FeatureSpec("pipeline_on_order", 0.0, 200.0),
        FeatureSpec("last_incoming_order", 0.0, 200.0),
        FeatureSpec("base_stock", 0.0, 200.0),
        FeatureSpec("inventory_position", -100.0, 200.0),
    )

    def __init__(
        self,
        *,
        model: Optional[TinyRecursiveModel] = None,
        feature_specs: Sequence[FeatureSpec] | None = None,
        history_length: int = 32,
        min_history: int | None = None,
        device: str | torch.device = "cpu",
        seq_len: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        order_floor: int = 0,
    ) -> None:
        features = feature_specs or self.DEFAULT_FEATURES
        self.encoder = ObservationEncoder(features)
        self.history = TRMHistoryBuffer(self.encoder, max_history=history_length)
        self.min_history = int(min_history) if min_history is not None else max(4, history_length // 4)
        self.device = torch.device(device)
        self.order_floor = int(order_floor)

        expected_tokens = history_length * self.encoder.feature_count
        self.seq_len = int(seq_len or expected_tokens)
        if self.seq_len < expected_tokens:
            raise ValueError(
                f"seq_len ({self.seq_len}) must be >= history_length * feature_count ({expected_tokens})."
            )

        self.model = model or build_default_trm_model(seq_len=self.seq_len)
        self.model.eval()
        self.model.to(self.device)

        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state)

        self._pad_token = self.encoder.pad_token

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def order(self, obs: Dict[str, Any]) -> int:
        """Return the order quantity for the current Beer Game timestep."""

        self.history.append(obs)

        if len(self.history) < self.min_history:
            return self._fallback_order(obs)

        sequence = self._build_sequence()
        with torch.no_grad():
            pred_tokens, _ = self.model.predict(sequence)

        # Use the final feature of the most recent timestep (incoming order)
        feature_index = self.encoder.feature_count - 1
        token_index = -self.encoder.feature_count + feature_index
        predicted_token = int(pred_tokens[token_index].item())
        predicted_value = self.encoder.decode_feature(feature_index, predicted_token)

        backlog = float(obs.get("backlog", 0.0) or 0.0)
        order_quantity = predicted_value + backlog

        return max(self.order_floor, int(round(order_quantity)))

    def reset(self) -> None:
        """Forget the accumulated history."""

        self.history.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fallback_order(self, obs: Dict[str, Any]) -> int:
        """Fallback policy while the TRM is warming up with history."""

        last_demand = float(obs.get("last_incoming_order", 0.0) or 0.0)
        backlog = float(obs.get("backlog", 0.0) or 0.0)
        base_stock = float(obs.get("base_stock", 0.0) or 0.0)
        inventory_position = float(obs.get("inventory_position", base_stock))

        correction = max(base_stock - inventory_position, 0.0)
        order_qty = last_demand + backlog + correction
        return max(self.order_floor, int(round(order_qty)))

    def _build_sequence(self) -> Tensor:
        tokens = self.history.tokens()
        if len(tokens) >= self.seq_len:
            tokens = tokens[-self.seq_len :]
        else:
            padding = [self._pad_token] * (self.seq_len - len(tokens))
            tokens = padding + tokens
        tensor = torch.tensor(tokens, dtype=torch.long, device=self.device)
        return tensor.unsqueeze(0)


__all__ = [
    "FeatureSpec",
    "ObservationEncoder",
    "TRMHistoryBuffer",
    "TRMSupplyPolicy",
    "build_default_trm_model",
]
