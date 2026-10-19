"""Optional real SmolVLA loading with checkpoint processors and output adapters.

Inputs are checkpoint-compatible batched tensors, not raw serial/camera values.
This module does not convert between radians, degrees, or normalized motor units.
"""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SensorSnapshot:
    timestamp: float
    state: np.ndarray
    batch: dict

    def __post_init__(self):
        if not np.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("Need a nonnegative monotonic observation timestamp")
        s = np.array(self.state, dtype=float, copy=True)
        if s.ndim != 1 or not len(s) or not np.isfinite(s).all():
            raise ValueError("Need a finite robot state")
        s.setflags(write=False)
        object.__setattr__(self, "state", s)
        object.__setattr__(self, "batch", deepcopy(self.batch))


class SmolVLAStream:
    def __init__(self, policy, preprocessor, postprocessor, *, adapter=None, checkpoint="user-supplied"):
        self.base, self.pre, self.post, self.checkpoint = policy, preprocessor, postprocessor, checkpoint
        if adapter is not None:
            if (getattr(getattr(policy, "config", None), "use_relative_actions", False)
                or any(type(step).__name__=="RelativeActionsProcessorStep" for step in getattr(preprocessor, "steps", []))):
                raise ValueError("The existing residual adapter contract requires absolute processed joint targets")
            from .adapter import AdaptedChunkPolicy
            self.policy = AdaptedChunkPolicy(policy, adapter)
        else:
            self.policy = policy

    @classmethod
    def from_pretrained(cls, checkpoint, *, device="cpu", adapter_path=None):
        from importlib.metadata import version
        if version("lerobot") != "0.6.1":
            raise RuntimeError("This bridge targets LeRobot 0.6.1")
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        policy = SmolVLAPolicy.from_pretrained(checkpoint).to(device).eval()
        pre, post = make_pre_post_processors(policy.config, pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}})
        adapter = None
        if adapter_path is not None:
            from .adapter import load_adapter
            adapter, metadata = load_adapter(adapter_path, device)
            if metadata.get("cache_metadata", {}).get("action_space") != "upstream_processed_absolute_joint_targets":
                raise ValueError("Adapter action-space declaration does not match")
        return cls(policy, pre, post, adapter=adapter, checkpoint=checkpoint)

    def reset(self):
        self.policy.reset()
        for processor in (self.pre, self.post):
            if callable(getattr(processor, "reset", None)):
                processor.reset()

    def __call__(self, snapshot):
        import torch
        with torch.inference_mode():
            batch = self.pre(deepcopy(snapshot.batch))
            state = batch["observation.state"]
            if state.ndim != 2 or state.shape[0] != 1:
                raise ValueError("Expected one preprocessed observation")
            actions = self.policy.predict_action_chunk(batch)
            if actions.ndim != 3 or actions.shape[0] != 1 or not 0 < actions.shape[1] <= 512:
                raise ValueError("Expected one nonempty action chunk")
            # Process each temporal action as a normal single-batch policy action.
            restored = torch.stack([self.post(actions[:, i, :]) for i in range(actions.shape[1])], dim=1)
            result = restored[0].detach().cpu().numpy()
            if result.ndim != 2 or result.shape[1] != len(snapshot.state) or not np.isfinite(result).all():
                raise ValueError("Physical action shape or values do not match robot state")
            return result.copy()


def async_runner(stream, buffer, *, sample_dt, clock):
    from .runtime import AsyncInference
    return AsyncInference(stream, buffer, sample_dt, clock=clock,
        snapshot_factory=lambda obs: SensorSnapshot(obs.timestamp, obs.state, obs.batch))


def ros_chunk(plan, joint_names, *, observation_ros_seconds):
    """Translate runtime.ActionChunk into the separately built ROS2 message.

    Explicitly convert the monotonic timestamp to the same ROS clock as the
    JointState publisher at the calling boundary. No clock offset is guessed.
    """
    from so101_control.msg import ActionChunk
    from trajectory_msgs.msg import JointTrajectoryPoint
    if (not np.isfinite(observation_ros_seconds) or observation_ros_seconds < 0
        or len(set(joint_names)) != len(joint_names) or not all(joint_names)
        or plan.actions.shape[1] != len(joint_names)):
        raise ValueError("Invalid joint names, dimensions or ROS timestamp")
    message = ActionChunk()
    message.sequence = plan.request_id
    nanoseconds = round(observation_ros_seconds * 1e9)
    message.observation_time.sec, message.observation_time.nanosec = divmod(nanoseconds, 1_000_000_000)
    message.sample_dt = plan.sample_dt
    message.joint_names = list(joint_names)
    message.reference_state = plan.reference_state.tolist()
    for values in plan.actions:
        point = JointTrajectoryPoint()
        point.positions = values.tolist()
        message.points.append(point)
    return message
