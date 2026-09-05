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
    probability_margin: float = 0.0
    known_gesture_mass: float = 1.0


@dataclass(slots=True)
class TemporalGate:
    config: RuntimeConfig
    max_history: int = 30
    _history: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=30))
    _label_started_at: float | None = None
    _last_label: str | None = None
    _rejected_frames: int = 0

    def reset(self) -> None:
        self._history.clear()
        self._label_started_at = None
        self._last_label = None
        self._rejected_frames = 0

    def reject(
        self,
        reason: str,
        *,
        probabilities: np.ndarray | None = None,
        known_gesture_mass: float = 0.0,
    ) -> TemporalDecision:
        self._rejected_frames += 1
        # Do not let a stale high-confidence EMA survive an invalid/no-hand pose.
        self._history.clear()
        self._label_started_at = None
        self._last_label = None
        values = (
            np.asarray(probabilities, dtype=np.float64).reshape(-1)
            if probabilities is not None
            else np.zeros(len(self.config.class_names), dtype=np.float64)
        )
        if len(values) != len(self.config.class_names):
            values = np.zeros(len(self.config.class_names), dtype=np.float64)
        total = float(values.sum())
        if total > 1e-12:
            values = values / total
        return TemporalDecision(
            execute=False,
            predicted_gesture=self.config.reject_label,
            confidence=0.0,
            stable_frames=0,
            frames_used=0,
            reason=reason,
            probabilities={
                name: float(values[index])
                for index, name in enumerate(self.config.class_names)
            },
            held_seconds=0.0,
            probability_margin=0.0,
            known_gesture_mass=float(max(0.0, min(1.0, known_gesture_mass))),
        )

    def update(
        self,
        probabilities: np.ndarray,
        now: float | None = None,
        *,
        known_gesture_mass: float = 1.0,
        pose_valid: bool = True,
        rejection_reason: str | None = None,
    ) -> TemporalDecision:
        now = time.monotonic() if now is None else float(now)
        row = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if len(row) != len(self.config.class_names):
            raise ValueError("Probability count does not match the configured class order.")
        if (
            not np.isfinite(row).all()
            or (row < 0.0).any()
            or float(row.sum()) <= 1e-12
        ):
            return self.reject("invalid classifier probabilities")
        if not np.isfinite(float(known_gesture_mass)):
            return self.reject("invalid known-gesture mass")
        row = np.clip(row, 1e-12, None)
        row /= row.sum()
        order = np.sort(row)
        raw_confidence = float(order[-1])
        probability_margin = float(order[-1] - order[-2]) if len(order) > 1 else raw_confidence
        if not pose_valid:
            return self.reject(
                rejection_reason or "pose geometry rejected",
                probabilities=row,
                known_gesture_mass=known_gesture_mass,
            )
        if known_gesture_mass < self.config.known_mass_floor:
            return self.reject(
                "outside the eight-gesture vocabulary",
                probabilities=row,
                known_gesture_mass=known_gesture_mass,
            )
        if raw_confidence < self.config.confidence_floor:
            return self.reject(
                "below confidence floor",
                probabilities=row,
                known_gesture_mass=known_gesture_mass,
            )
        if probability_margin < self.config.probability_margin_floor:
            return self.reject(
                "ambiguous probability margin",
                probabilities=row,
                known_gesture_mass=known_gesture_mass,
            )
        self._rejected_frames = 0
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
        label_started_at = (
            now if self._label_started_at is None else self._label_started_at
        )
        held_seconds = max(0.0, now - label_started_at)
        required_hold = self.config.minimum_hold_seconds
        execute = bool(
            confidence >= self.config.confidence_floor
            and stable_frames >= self.config.stable_frames_required
            and held_seconds >= required_hold
        )
        if confidence < self.config.confidence_floor:
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
            probability_margin=probability_margin,
            known_gesture_mass=float(max(0.0, min(1.0, known_gesture_mass))),
        )
