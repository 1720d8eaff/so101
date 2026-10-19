"""Causal history windows, explicit failure supervision and train-only sampling."""
from dataclasses import dataclass
import numpy as np

FAILURE_CLASSES = ("none", "localization", "grasp", "execution_deviation", "target_change",
                   "early_stop", "timeout", "collision", "other")


def matrix(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.ndim != 2 or 0 in array.shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite, nonempty [T,D]")
    return array.astype(np.float32)


def history_windows(states, length):
    """One episode per call; left zero padding is masked, future frames excluded."""
    states = matrix(states, "states")
    if type(length) is not int or length < 1:
        raise ValueError("history length must be a positive integer")
    windows = np.zeros((len(states), length, states.shape[1]), dtype=np.float32)
    mask = np.zeros((len(states), length), dtype=bool)
    for t in range(len(states)):
        n = min(length, t + 1)
        windows[t, -n:] = states[t - n + 1:t + 1]
        mask[t, -n:] = True
    return windows, mask


def action_targets(actions, horizon):
    """Future actions are supervision, never inputs; episode boundaries are padded."""
    actions = matrix(actions, "actions")
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    targets = np.zeros((len(actions), horizon, actions.shape[1]), dtype=np.float32)
    mask = np.zeros((len(actions), horizon), dtype=bool)
    for t in range(len(actions)):
        n = min(horizon, len(actions) - t)
        targets[t, :n] = actions[t:t + n]
        mask[t, :n] = True
    return targets, mask


@dataclass(frozen=True)
class FailureEvent:
    start: int
    stop: int
    failure_type: str
    recovered: bool | None = None

    def __post_init__(self):
        if (type(self.start) is not int or type(self.stop) is not int
            or not 0 <= self.start < self.stop or self.failure_type not in FAILURE_CLASSES[1:]
            or (self.recovered is not None and type(self.recovered) is not bool)):
            raise ValueError("Invalid annotated failure event; intervals are [start,stop)")


def failure_labels(frames, events, verified_no_failure=False):
    """Unknown frames remain -1 unless explicitly annotated as no failure.

    Labels are training targets. This function does not infer that an
    unannotated frame is a success, or feed human labels to online inference.
    """
    if type(frames) is not int or frames < 1 or type(verified_no_failure) is not bool:
        raise ValueError("Invalid frame count or annotation flag")
    labels = np.full(frames, 0 if verified_no_failure else -1, dtype=np.int64)
    occupied = np.zeros(frames, dtype=bool)
    for event in events:
        if event.stop > frames or occupied[event.start:event.stop].any():
            raise ValueError("Failure events overlap or exceed episode")
        labels[event.start:event.stop] = FAILURE_CLASSES.index(event.failure_type)
        occupied[event.start:event.stop] = True
    return labels


def recovery_windows(frames, events, before=15, after=30):
    if any(type(v) is not int or v < 0 for v in (before, after)) or type(frames) is not int or frames < 1:
        raise ValueError("Invalid recovery window parameters")
    failure_labels(frames, events)  # Validate bounds and overlaps first.
    intervals = sorted((max(0, e.start - before), min(frames, e.stop + after)) for e in events)
    merged = []
    for start, stop in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return merged


def training_weights(episode_ids, labels, split, boost=3.):
    """Return only training indices and hard-case weights; never sample validation."""
    ids = np.asarray(episode_ids)
    labels = np.asarray(labels)
    if ids.ndim != 1 or ids.dtype.kind not in "iuUS" or not len(ids) or labels.shape != ids.shape or labels.dtype.kind not in "iu":
        raise ValueError("Expected aligned episode IDs and integer labels")
    ids = ids.astype(str)
    train, validation = [str(v) for v in split["train"]], [str(v) for v in split["validation"]]
    if (not train or not validation or len(set(train)) != len(train) or len(set(validation)) != len(validation)
        or set(train) & set(validation) or set(train + validation) != set(ids)):
        raise ValueError("Split must cover all episodes exactly, without train/validation overlap")
    if (not np.isfinite(boost) or boost < 1 or np.any(labels < -1) or np.any(labels >= len(FAILURE_CLASSES))):
        raise ValueError("Invalid hard-case boost or failure label")
    indices = np.flatnonzero(np.isin(ids, train))
    weights = np.where(labels[indices] > 0, boost, 1.).astype(np.float64)
    return indices, weights
