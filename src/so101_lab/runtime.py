"""Bounded asynchronous inference and timestamp/state-aligned joint targets.

Pure scheduling code: it does not communicate with a robot. All timestamps
must use one monotonic clock; actions and joint limits use the same units.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import threading
import time
import numpy as np


def vector(value, name):
    out = np.array(value, dtype=np.float64, copy=True)
    if out.ndim != 1 or not out.size or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    return out


def timestamp(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Timestamps must be finite nonnegative monotonic seconds")
    return float(value)


@dataclass(frozen=True)
class Observation:
    timestamp: float
    state: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "timestamp", timestamp(self.timestamp))
        state = vector(self.state, "state")
        state.setflags(write=False)
        object.__setattr__(self, "state", state)


@dataclass(frozen=True)
class ActionChunk:
    request_id: int
    observation_timestamp: float
    generated_timestamp: float
    sample_dt: float
    reference_state: np.ndarray
    actions: np.ndarray

    def __post_init__(self):
        if type(self.request_id) is not int or self.request_id < 0:
            raise ValueError("request_id must be a nonnegative integer")
        obs, generated = timestamp(self.observation_timestamp), timestamp(self.generated_timestamp)
        if generated < obs or isinstance(self.sample_dt, bool) or not math.isfinite(self.sample_dt) or self.sample_dt <= 0:
            raise ValueError("Invalid action period or ordered timestamps")
        actions = np.array(self.actions, dtype=np.float64, copy=True)
        if actions.ndim != 2 or 0 in actions.shape or not np.isfinite(actions).all():
            raise ValueError("actions must be finite nonempty [horizon, action_dim]")
        state = vector(self.reference_state, "reference_state")
        state.setflags(write=False)
        actions.setflags(write=False)
        object.__setattr__(self, "reference_state", state)
        object.__setattr__(self, "actions", actions)


@dataclass(frozen=True)
class BufferSettings:
    max_observation_age_s: float = .5
    max_action_lag_s: float = .07
    max_reference_error: float = .25
    max_pending_steps: int = 64
    replan_below_steps: int = 8

    def __post_init__(self):
        for value in (self.max_observation_age_s, self.max_action_lag_s, self.max_reference_error):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("Buffer thresholds must be finite and positive")
        if (type(self.max_pending_steps) is not int or type(self.replan_below_steps) is not int
            or self.max_pending_steps < 1 or not 0 <= self.replan_below_steps < self.max_pending_steps):
            raise ValueError("Invalid buffer capacity / replan threshold")


@dataclass(frozen=True)
class ControlStep:
    action: np.ndarray
    status: str
    request_replan: bool
    pending_steps: int


class StateAlignedBuffer:
    """Absolute targets: sample k is due at observation_time + (k+1)*dt.

    A new chunk replaces only future scheduled targets. Late prefixes and
    out-of-order requests are discarded. State drift is an infinity norm in
    caller-specified state units. Continuity is bounded per control tick.
    """
    def __init__(self, initial_action, lower, upper, max_step, settings=None):
        self.last_action = vector(initial_action, "initial_action")
        self.lower, self.upper = vector(lower, "lower"), vector(upper, "upper")
        self.max_step = np.broadcast_to(np.asarray(max_step, dtype=float), self.last_action.shape).copy()
        if (self.lower.shape != self.last_action.shape or self.upper.shape != self.last_action.shape
            or np.any(self.lower >= self.upper) or np.any(self.last_action < self.lower)
            or np.any(self.last_action > self.upper) or not np.isfinite(self.max_step).all()
            or np.any(self.max_step <= 0)):
            raise ValueError("Invalid limits, initial target or per-tick step bound")
        self.settings = settings or BufferSettings()
        self.queue, self.last_request, self.last_tick = [], -1, None
        self.counters = Counter()
        self.lock = threading.Lock()

    def submit(self, chunk, now, current_state):
        now = timestamp(now)
        state = vector(current_state, "current_state")
        c = self.settings
        with self.lock:
            if self.last_tick is not None and now < self.last_tick:
                raise ValueError("Clock moved backwards")
            if chunk.actions.shape[1:] != self.last_action.shape or state.shape != chunk.reference_state.shape:
                raise ValueError("State/action dimensions changed")
            if chunk.generated_timestamp > now + 1e-9 or chunk.observation_timestamp > now:
                raise ValueError("Chunk is from a future clock")
            reason = None
            if chunk.request_id <= self.last_request:
                reason = "out_of_order"
            elif now - chunk.observation_timestamp > c.max_observation_age_s:
                reason = "stale_observation"
            elif np.max(np.abs(state - chunk.reference_state)) > c.max_reference_error:
                reason = "state_drift"
            incoming = [(chunk.observation_timestamp + (i + 1) * chunk.sample_dt, action.copy(), chunk.observation_timestamp)
                        for i, action in enumerate(chunk.actions)
                        if chunk.observation_timestamp + (i + 1) * chunk.sample_dt >= now - 1e-9]
            if not incoming and reason is None:
                reason = "expired_chunk"
            if reason:
                self.counters[reason] += 1
                return reason
            # Keep an existing prefix before the first new timestamp. Never
            # append two chunks end to end after an inference delay.
            prefix = [item for item in self.queue if now <= item[0] < incoming[0][0] - 1e-9]
            self.queue = (prefix + incoming)[:c.max_pending_steps]
            self.last_request = chunk.request_id
            self.counters["accepted"] += 1
            self.counters["discarded_prefix_steps"] += len(chunk.actions) - len(incoming)
            return "accepted"

    def tick(self, now, observation_timestamp):
        now, observation_timestamp = timestamp(now), timestamp(observation_timestamp)
        with self.lock:
            if (self.last_tick is not None and now <= self.last_tick) or observation_timestamp > now:
                raise ValueError("Control time must increase and observations cannot be in the future")
            self.last_tick = now
            status, selected = "hold", None
            if now - observation_timestamp > self.settings.max_observation_age_s:
                self.queue.clear()
                status = "stale_observation"
            else:
                valid_queue = [item for item in self.queue if now - item[2] <= self.settings.max_observation_age_s]
                self.counters["expired_at_execution"] += len(self.queue) - len(valid_queue)
                self.queue = valid_queue
                while self.queue and self.queue[0][0] <= now + 1e-9:
                    item = self.queue.pop(0)
                    if (now - item[0] <= self.settings.max_action_lag_s
                        and now - item[2] <= self.settings.max_observation_age_s):
                        selected = item[1]
                    else:
                        self.counters["expired_at_execution"] += 1
                if selected is not None:
                    bounded = np.clip(selected, self.lower, self.upper)
                    delta = np.clip(bounded - self.last_action, -self.max_step, self.max_step)
                    self.last_action = self.last_action + delta
                    status = "execute"
            self.counters[status] += 1
            return ControlStep(self.last_action.copy(), status,
                               len(self.queue) <= self.settings.replan_below_steps, len(self.queue))


class AsyncInference:
    """One in-flight prediction; ``advance`` never waits for GPU inference.

    ``predict(Observation)`` returns absolute targets [H,A]. History/image
    preprocessing belongs in that callable. No middleware or robot is opened.
    """
    def __init__(self, predict, buffer, sample_dt, clock=time.monotonic, snapshot_factory=None):
        if not math.isfinite(sample_dt) or sample_dt <= 0:
            raise ValueError("Invalid action sampling period")
        self.predict, self.buffer, self.sample_dt, self.clock = predict, buffer, sample_dt, clock
        self.snapshot_factory = snapshot_factory or (lambda obs: Observation(obs.timestamp, obs.state))
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="so101-policy")
        self.future, self.next_id, self.closed, self.last_error = None, 0, False, None

    def advance(self, observation, now=None):
        if self.closed:
            raise RuntimeError("Inference runner is closed")
        now = timestamp(self.clock() if now is None else now)
        if self.future is not None and self.future.done():
            try:
                self.buffer.submit(self.future.result(), now, observation.state)
                self.last_error = None
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.buffer.counters["policy_error"] += 1
            finally:
                self.future = None
        step = self.buffer.tick(now, observation.timestamp)
        if (step.request_replan and self.future is None
            and 0 <= now - observation.timestamp <= self.buffer.settings.max_observation_age_s):
            request_id, snapshot = self.next_id, self.snapshot_factory(observation)
            self.next_id += 1
            def infer():
                actions = self.predict(snapshot)
                return ActionChunk(request_id, snapshot.timestamp, self.clock(), self.sample_dt, snapshot.state, actions)
            self.future = self.pool.submit(infer)
        return step

    def close(self, wait=True):
        self.closed = True
        self.pool.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
