"""History/failure conditioning inside SmolVLA's action-time embedding path.

Targets supervise auxiliary heads only. No extra sequence tokens are inserted,
so the original attention mask, chunk length, and zero-initialized model output
are preserved. The integration targets LeRobot 0.6.1 action_time_mlp_out.
"""
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .recovery import FAILURE_CLASSES


@dataclass(frozen=True)
class InternalConfig:
    state_dim: int
    expert_dim: int
    history: int = 4
    hidden_dim: int = 64


class InternalConditioner(nn.Module):
    def __init__(self, config):
        super().__init__()
        if any(type(v) is not int or v<1 for v in asdict(config).values()):raise ValueError("Invalid internal adapter dimensions")
        self.config=config
        self.encoder=nn.Sequential(nn.Linear(config.history*(config.state_dim+1),config.hidden_dim),nn.SiLU())
        self.failure=nn.Linear(config.hidden_dim,len(FAILURE_CLASSES));self.progress=nn.Linear(config.hidden_dim,1)
        self.residual=nn.Linear(config.hidden_dim+len(FAILURE_CLASSES)+1,config.expert_dim)
        nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)

    def forward(self, history, mask):
        c=self.config
        if (history.ndim!=3 or history.shape[1:]!=(c.history,c.state_dim) or mask.shape!=history.shape[:2]
            or mask.dtype!=torch.bool or not mask[:,-1].all() or not torch.isfinite(history).all()):
            raise ValueError("Invalid causal history or mask")
        history=history.to(dtype=self.encoder[0].weight.dtype)
        hidden=self.encoder(torch.cat([(history*mask.unsqueeze(-1)).flatten(1),mask.to(history.dtype)],dim=1))
        failure=self.failure(hidden);progress=torch.sigmoid(self.progress(hidden)).squeeze(-1)
        residual=self.residual(torch.cat([hidden,torch.softmax(failure,dim=-1),progress[:,None]],dim=1))
        return dict(residual=residual,failure_logits=failure,progress=progress)


class InternalBinding:
    def __init__(self, policy, conditioner):
        if getattr(getattr(policy,"config",None),"compile_model",False):raise ValueError("Disable compile_model before installing hooks")
        layer=policy.model.action_time_mlp_out
        if not isinstance(layer,nn.Linear) or layer.out_features!=conditioner.config.expert_dim:
            raise ValueError("Expected LeRobot 0.6.1 action-time output projection")
        if hasattr(policy,"history_conditioner"):raise ValueError("An internal conditioner is already attached")
        self.context=ContextVar("smolvla_internal_history",default=None)
        self.policy,self.conditioner=policy,conditioner
        policy.add_module("history_conditioner",conditioner)
        self.handle=layer.register_forward_hook(self._hook)

    def _hook(self, layer, args, output):
        context=self.context.get()
        if context is None:raise RuntimeError("Internal conditioner needs an explicit causal history context")
        history,mask=context
        values=self.conditioner(history,mask)
        if output.ndim!=3 or output.shape[0]!=history.shape[0]:raise ValueError("Action-time embedding shape changed")
        return output+values["residual"].to(device=output.device,dtype=output.dtype)[:,None,:]

    @contextmanager
    def conditioned(self, history, mask):
        token=self.context.set((history,mask))
        try:yield
        finally:self.context.reset(token)

    def close(self):
        self.handle.remove()
        if hasattr(self.policy,"history_conditioner"):delattr(self.policy,"history_conditioner")


class InternalChunkPolicy:
    def __init__(self, base_policy, conditioner):
        self.base_policy=base_policy
        self.binding=InternalBinding(base_policy,conditioner)
        self.history=deque(maxlen=conditioner.config.history)

    def reset(self):
        self.history.clear();self.base_policy.reset()

    def predict_action_chunk(self,batch):
        state=batch["observation.state"];c=self.binding.conditioner.config
        if state.shape!=(1,c.state_dim):raise ValueError("Expected one normalized robot state")
        self.history.append(state[0].detach().clone())
        window=state.new_zeros((1,c.history,c.state_dim));mask=torch.zeros((1,c.history),dtype=torch.bool,device=state.device)
        window[0,-len(self.history):]=torch.stack(list(self.history));mask[0,-len(self.history):]=True
        with self.binding.conditioned(window,mask):return self.base_policy.predict_action_chunk(batch)


def training_loss(binding,batch,history,mask,*,failure_target=None,progress_target=None):
    with binding.conditioned(history,mask):
        result=binding.policy(batch)
    loss=result[0] if isinstance(result,tuple) else result
    if not isinstance(loss,torch.Tensor) or loss.numel()!=1 or not torch.isfinite(loss).all():
        raise ValueError("Base forward must return a finite scalar training loss")
    values=binding.conditioner(history,mask)
    auxiliary=loss*0
    if failure_target is not None:
        if failure_target.shape!=(len(history),) or failure_target.dtype!=torch.long or ((failure_target < -1)|(failure_target>=len(FAILURE_CLASSES))).any():
            raise ValueError("Invalid failure labels; -1 means unknown")
        valid=failure_target>=0
        if valid.any():auxiliary=auxiliary+.1*F.cross_entropy(values["failure_logits"][valid],failure_target[valid])
    if progress_target is not None:
        if progress_target.shape!=(len(history),) or not torch.isfinite(progress_target).all() or ((progress_target!=-1)&((progress_target<0)|(progress_target>1))).any():
            raise ValueError("Invalid progress labels")
        valid=progress_target>=0
        if valid.any():auxiliary=auxiliary+.1*F.mse_loss(values["progress"][valid],progress_target[valid])
    return loss+auxiliary


def save(path,conditioner,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as f:
        torch.save(dict(schema_version=1,config=asdict(conditioner.config),weights=conditioner.state_dict(),
            metadata=metadata,injection="action_time_mlp_out residual; not output-action residual"),f)


def load(path,device="cpu"):
    state=torch.load(path,map_location=device,weights_only=True)
    if state.get("schema_version")!=1:raise ValueError("Unsupported internal adapter schema")
    adapter=InternalConditioner(InternalConfig(**state["config"])).to(device)
    adapter.load_state_dict(state["weights"],strict=True)
    return adapter.eval(),state["metadata"]
