"""Trainable history / failure-aware residual on frozen policy action chunks.

This is a new action-output adapter, not a reproduction of an unpublished
SmolVLA token-injection experiment. Ground-truth failure/progress are targets
only; inference conditions on its own predicted heads.
"""
from collections import deque
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .recovery import FAILURE_CLASSES


@dataclass(frozen=True)
class AdapterConfig:
    state_dim: int
    action_dim: int
    history: int = 4
    horizon: int = 16
    hidden_dim: int = 64
    mode: str = "failure_aware"

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in
               (self.state_dim, self.action_dim, self.history, self.horizon, self.hidden_dim)):
            raise ValueError("Adapter dimensions must be positive integers")
        if self.mode not in {"history", "failure_aware"}:
            raise ValueError("mode must be history or failure_aware; original is the frozen base")


class HistoryFailureAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        self.history_encoder = nn.Sequential(nn.Linear(c.history * (c.state_dim + 1), c.hidden_dim), nn.SiLU(),
                                             nn.Linear(c.hidden_dim, c.hidden_dim), nn.SiLU())
        self.failure_head = nn.Linear(c.hidden_dim, len(FAILURE_CLASSES))
        self.progress_head = nn.Linear(c.hidden_dim, 1)
        extra = len(FAILURE_CLASSES) + 1 if c.mode == "failure_aware" else 0
        self.residual = nn.Sequential(nn.Linear(c.hidden_dim + extra + c.horizon * c.action_dim, c.hidden_dim),
                                      nn.SiLU(), nn.Linear(c.hidden_dim, c.horizon * c.action_dim))
        # The initial model exactly preserves frozen policy output.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, history_state, history_mask, base_actions):
        c, batch = self.config, history_state.shape[0]
        if (history_state.shape != (batch, c.history, c.state_dim)
            or history_mask.shape != (batch, c.history) or history_mask.dtype != torch.bool
            or base_actions.shape != (batch, c.horizon, c.action_dim) or batch == 0
            or not history_mask[:, -1].all()):
            raise ValueError("Invalid history, mask or base action shape")
        if not torch.isfinite(history_state).all() or not torch.isfinite(base_actions).all():
            raise ValueError("Nonfinite adapter input")
        masked = history_state * history_mask.unsqueeze(-1)
        hidden = self.history_encoder(torch.cat((masked.flatten(1), history_mask.to(history_state.dtype)), dim=1))
        logits, progress = self.failure_head(hidden), torch.sigmoid(self.progress_head(hidden)).squeeze(-1)
        features = [hidden, base_actions.flatten(1)]
        if c.mode == "failure_aware":
            features += [torch.softmax(logits, dim=-1), progress[:, None]]
        delta = self.residual(torch.cat(features, dim=1)).reshape(batch, c.horizon, c.action_dim)
        return dict(actions=base_actions + delta, failure_logits=logits, progress=progress)


def adapter_loss(output, target_actions, action_mask, failure_target=None, progress_target=None,
                 failure_weight=.1, progress_weight=.1):
    prediction = output["actions"]
    if (target_actions.shape != prediction.shape or action_mask.shape != prediction.shape[:2]
        or action_mask.dtype != torch.bool or not action_mask.any(dim=1).all()
        or not torch.isfinite(target_actions).all()):
        raise ValueError("Invalid target actions or action mask")
    if any(not math.isfinite(w) or w < 0 for w in (failure_weight, progress_weight)):
        raise ValueError("Loss weights must be nonnegative")
    error = (prediction - target_actions).square().mean(dim=-1)
    action = error[action_mask].mean()
    failure, progress = action * 0, action * 0
    if failure_target is not None:
        if (failure_target.shape != prediction.shape[:1] or failure_target.dtype != torch.long
            or (failure_target < -1).any() or (failure_target >= len(FAILURE_CLASSES)).any()):
            raise ValueError("Invalid failure targets; -1 means unknown")
        valid = failure_target >= 0
        if valid.any():
            failure = F.cross_entropy(output["failure_logits"][valid], failure_target[valid])
    if progress_target is not None:
        if (progress_target.shape != prediction.shape[:1] or not torch.isfinite(progress_target).all()
            or ((progress_target != -1) & ((progress_target < 0) | (progress_target > 1))).any()):
            raise ValueError("Invalid progress targets; -1 means unknown")
        valid = progress_target >= 0
        if valid.any():
            progress = F.mse_loss(output["progress"][valid], progress_target[valid])
    return dict(total=action + failure_weight * failure + progress_weight * progress,
                action=action, failure=failure, progress=progress)


class AdaptedChunkPolicy:
    """LeRobot-style ``predict_action_chunk`` wrapper, batch size one.

    The caller supplies the upstream preprocessed batch and postprocesses the
    returned normalized action. Use the same train-only normalization for
    cached base actions, targets and deployment. Call reset at each episode.
    """
    def __init__(self, base_policy, adapter):
        self.base_policy, self.adapter = base_policy, adapter.eval()
        self.history = deque(maxlen=adapter.config.history)

    def reset(self):
        self.history.clear()
        self.base_policy.reset()

    @torch.inference_mode()
    def predict_action_chunk(self, batch):
        state = batch["observation.state"]
        c = self.adapter.config
        if state.shape != (1, c.state_dim):
            raise ValueError("Wrapper expects one robot state per call")
        self.history.append(state[0].detach().clone())
        windows = state.new_zeros((1, c.history, c.state_dim))
        mask = torch.zeros((1, c.history), dtype=torch.bool, device=state.device)
        windows[0, -len(self.history):] = torch.stack(list(self.history))
        mask[0, -len(self.history):] = True
        base = self.base_policy.predict_action_chunk(batch)
        if base.ndim != 3 or base.shape[1] < c.horizon:
            raise ValueError("Base policy action horizon is too short")
        return self.adapter(windows, mask, base[:, :c.horizon])["actions"]


def save_adapter(path, model, metadata):
    path = Path(path)
    if path.exists():
        raise ValueError("Refusing to overwrite an adapter checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(dict(schema_version=1, config=asdict(model.config), state_dict=model.state_dict(),
                        metadata=metadata, failure_classes=list(FAILURE_CLASSES)), stream)


def load_adapter(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["schema_version"] != 1 or checkpoint["failure_classes"] != list(FAILURE_CLASSES):
        raise ValueError("Unsupported adapter checkpoint or label order")
    model = HistoryFailureAdapter(AdapterConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval(), checkpoint["metadata"]
