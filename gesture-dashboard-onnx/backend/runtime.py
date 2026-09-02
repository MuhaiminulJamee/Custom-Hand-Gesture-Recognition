from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time

import numpy as np

from .config import RuntimeConfig


def probability_ema(probability_rows: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    rows = np.asarray(probability_rows, dtype=np.float64)
    if rows.ndim != 2 or not len(rows):
        raise ValueError("EMA requires at least one probability row.")
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("EMA alpha must be in (0, 1].")
    output: list[np.ndarray] = []
    state = rows[0].copy()
    state /= max(float(state.sum()), 1e-12)
    output.append(state.copy())
    for row in rows[1:]:
        state = float(alpha) * row + (1.0 - float(alpha)) * state
        state = np.clip(state, 1e-12, None)
        state /= state.sum()
        output.append(state.copy())
    return np.vstack(output)


def trailing_equal_count(values: list[int] | np.ndarray) -> int:
    values = list(values)
    if not values:
        return 0
    final_value = values[-1]
    count = 0
    for value in reversed(values):
        if value != final_value:
            break
        count += 1
    return count


@dataclass(slots=True)
class TemporalDecision:
    execute: bool
    predicted_gesture: str
    confidence: float
    stable_frames: int
    frames_used: int
    reason: str
    probabilities: dict[str, float]
    held_seconds: float


@dataclass(slots=True)
class TemporalGate:
    config: RuntimeConfig
    max_history: int = 30
    _history: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=30))
    _label_started_at: float | None = None
    _last_label: str | None = None

    def reset(self) -> None:
        self._history.clear()
        self._label_started_at = None
        self._last_label = None

    def update(self, probabilities: np.ndarray, now: float | None = None) -> TemporalDecision:
        now = time.monotonic() if now is None else float(now)
        row = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if len(row) != len(self.config.class_names):
            raise ValueError("Probability count does not match the configured class order.")
        row = np.clip(row, 1e-12, None)
        row /= row.sum()
        self._history.append(row)
        ema_rows = probability_ema(np.vstack(self._history), self.config.ema_alpha)
        labels = ema_rows.argmax(axis=1)
        final_index = int(labels[-1])
        gesture = self.config.class_names[final_index]
        confidence = float(ema_rows[-1, final_index])
        stable_frames = trailing_equal_count(labels)

        if gesture != self._last_label:
            self._last_label = gesture
            self._label_started_at = now
        held_seconds = max(0.0, now - (self._label_started_at or now))
        required_hold = self.config.zoom_hold_seconds if gesture in {"zoom_in", "zoom_out"} else 0.0
        execute = bool(
            gesture != "no_gesture"
            and confidence >= self.config.confidence_floor
            and stable_frames >= self.config.stable_frames_required
            and held_seconds >= required_hold
        )
        if gesture == "no_gesture":
            reason = "negative class"
        elif confidence < self.config.confidence_floor:
            reason = "below confidence floor"
        elif stable_frames < self.config.stable_frames_required:
            reason = "insufficient consecutive stable frames"
        elif held_seconds < required_hold:
            reason = "gesture hold requirement not reached"
        else:
            reason = "stable and confident"
        return TemporalDecision(
            execute=execute,
            predicted_gesture=gesture,
            confidence=confidence,
            stable_frames=stable_frames,
            frames_used=len(ema_rows),
            reason=reason,
            probabilities={
                name: float(ema_rows[-1, index])
                for index, name in enumerate(self.config.class_names)
            },
            held_seconds=held_seconds,
        )
