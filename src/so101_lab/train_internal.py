"""Train internal conditioning with the frozen base policy's real flow loss.

The caller supplies checkpoint-processed batches, images and causal history.
Unlike output-cache training this calls the base model each optimization step.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import re
import torch
from .internal_conditioning import InternalBinding,save,training_loss


@dataclass
class Example:
    episode_id: str
    batch: dict
    history: torch.Tensor
    mask: torch.Tensor
    failure: torch.Tensor | None = None
    progress: torch.Tensor | None = None


def train(policy,conditioner,examples,split,metadata,output,*,epochs=20,lr=1e-3,seed=0):
    examples=list(examples);output=Path(output)
    train_ids=set(split["train"]);val_ids=set(split["validation"])
    if (not train_ids or not val_ids or train_ids&val_ids or not examples
        or {e.episode_id for e in examples}!=train_ids|val_ids):raise ValueError("Incomplete or overlapping episode split")
    if metadata.get("source") not in {"real_robot","simulation","synthetic"}:raise ValueError("Explicit source required")
    if set(metadata.get("normalization_fit_episodes",[]))!=train_ids:raise ValueError("Train-only preprocessing declaration required")
    if not re.fullmatch(r"[0-9a-f]{64}",metadata.get("checkpoint_sha256","")):raise ValueError("Record the exact base checkpoint hash")
    if (type(epochs) is not int or epochs<1 or type(seed) is not int or seed<0
        or lr<=0 or not torch.isfinite(torch.tensor(lr))):raise ValueError("Invalid training options")
    if output.exists():raise FileExistsError("Refusing to overwrite a training run")
    for e in examples:
        state=e.batch.get("observation.state")
        if (not isinstance(state,torch.Tensor) or state.ndim!=2 or e.history.ndim!=3
            or e.history.shape[0]!=state.shape[0] or not torch.equal(e.history[:,-1],state)):
            raise ValueError("History must end at the current processed state")
    original_grad={name:p.requires_grad for name,p in policy.named_parameters()}
    original_training=policy.training
    binding=InternalBinding(policy,conditioner)
    policy.requires_grad_(False);policy.eval();conditioner.requires_grad_(True)
    generator=torch.Generator().manual_seed(seed)
    def loss(e):
        return training_loss(binding,e.batch,e.history,e.mask,failure_target=e.failure,progress_target=e.progress)
    def evaluate(group):
        with torch.no_grad():return sum(float(loss(e)) for e in group)/len(group)
    train_rows=[e for e in examples if e.episode_id in train_ids]
    val_rows=[e for e in examples if e.episode_id in val_ids]
    history=[]
    try:
        # Validate every row through the actual model before creating outputs.
        for e in examples:
            with torch.no_grad():loss(e)
        optimizer=torch.optim.AdamW(conditioner.parameters(),lr=lr)
        for epoch in range(epochs):
            conditioner.train()
            for index in torch.randperm(len(train_rows),generator=generator).tolist():
                optimizer.zero_grad(set_to_none=True);value=loss(train_rows[index]);value.backward()
                torch.nn.utils.clip_grad_norm_(conditioner.parameters(),1.);optimizer.step()
            conditioner.eval()
            history.append(dict(epoch=epoch+1,train_loss=evaluate(train_rows),validation_loss=evaluate(val_rows)))
        output.mkdir(parents=True,exist_ok=False)
        run=dict(source=metadata["source"],metadata=metadata,split=split,epochs=epochs,lr=lr,seed=seed,
            base_frozen=True,base_checkpoint_included=False,injection="action_time_mlp_out",
            torch_version=str(torch.__version__),metric="flow loss plus auxiliary heads; not task success")
        save(output/"internal-adapter.pt",conditioner,run)
        for name,data in (("run.json",run),("losses.json",history)):
            with (output/name).open("x",encoding="utf-8") as f:json.dump(data,f,indent=2,allow_nan=False)
    finally:
        binding.close()
        for name,p in policy.named_parameters():p.requires_grad_(original_grad[name])
        policy.train(original_training)
    return history
