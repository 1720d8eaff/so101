"""Train a residual adapter on cached frozen-policy outputs, split by episode."""
import argparse
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
import numpy as np
from .io import read_json, write_json
from .recovery import training_weights


def train(cache_path, split_path, metadata_path, output_dir, *, mode="failure_aware", epochs=20,
          batch_size=32, learning_rate=1e-3, seed=0):
    import torch
    from .adapter import AdapterConfig, HistoryFailureAdapter, adapter_loss, save_adapter
    if (type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1
        or type(seed) is not int or seed < 0 or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("Invalid training hyperparameters")
    output = Path(output_dir)
    if output.exists():
        raise ValueError("Refusing to overwrite a training run")
    cache_bytes = Path(cache_path).read_bytes()
    split_bytes, metadata_bytes = Path(split_path).read_bytes(), Path(metadata_path).read_bytes()
    split, metadata = json.loads(split_bytes), json.loads(metadata_bytes)
    if not re.fullmatch(r"[0-9a-f]{64}", metadata.get("base_checkpoint_sha256", "")):
        raise ValueError("Metadata needs the frozen base checkpoint SHA256")
    if (sorted(map(str, metadata.get("normalization_training_episodes", []))) != sorted(map(str, split["train"]))
        or metadata.get("action_space") != "upstream_processed_absolute_joint_targets"):
        raise ValueError("Cache normalization IDs or action convention do not match the declared split")
    with np.load(BytesIO(cache_bytes), allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    required = {"history_state", "history_mask", "base_actions", "target_actions", "action_mask",
                "failure_target", "progress_target", "episode_id"}
    if set(arrays) != required:
        raise ValueError(f"Cache keys must be exactly {sorted(required)}")
    h, base, targets = arrays["history_state"], arrays["base_actions"], arrays["target_actions"]
    if h.ndim != 3 or base.ndim != 3 or 0 in h.shape or 0 in base.shape or len(h) != len(base) or base.shape != targets.shape:
        raise ValueError("Invalid cached batch shapes")
    if any(value.ndim == 0 or len(value) != len(h) for value in arrays.values()):
        raise ValueError("All cached arrays must have the same example count")
    if arrays["history_mask"].dtype != bool or arrays["action_mask"].dtype != bool:
        raise ValueError("Padding masks must be boolean")
    indices, weights = training_weights(arrays["episode_id"], arrays["failure_target"], split)
    ids = arrays["episode_id"].astype(str)
    validation = np.flatnonzero(np.isin(ids, list(map(str, split["validation"]))))
    tensors = {}
    for key, value in arrays.items():
        if key == "episode_id":
            continue
        if key.endswith("mask"):
            tensors[key] = torch.from_numpy(value).bool()
        elif key == "failure_target":
            tensors[key] = torch.from_numpy(value).long()
        else:
            if value.dtype.kind not in "iuf" or not np.isfinite(value).all():
                raise ValueError(f"Nonfinite or invalid cache values: {key}")
            tensors[key] = torch.from_numpy(value.astype(np.float32))
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    config = AdapterConfig(state_dim=h.shape[2], action_dim=base.shape[2], history=h.shape[1], horizon=base.shape[1], mode=mode)
    model = HistoryFailureAdapter(config)
    def loss_for(index):
        batch = {k: v[index] for k, v in tensors.items()}
        prediction = model(batch["history_state"], batch["history_mask"], batch["base_actions"])
        return adapter_loss(prediction, batch["target_actions"], batch["action_mask"],
                            batch["failure_target"], batch["progress_target"],
                            failure_weight=.1 if mode == "failure_aware" else 0.,
                            progress_weight=.1 if mode == "failure_aware" else 0.)
    # Validate ALL examples before creating an output directory or optimizing.
    with torch.no_grad():
        for start in range(0, len(h), batch_size):
            loss_for(slice(start, start + batch_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(epochs):
        model.train()
        selected = torch.multinomial(torch.as_tensor(weights), len(indices), replacement=True, generator=generator).numpy()
        sampled = indices[selected]
        for start in range(0, len(sampled), batch_size):
            optimizer.zero_grad(set_to_none=True)
            losses = loss_for(sampled[start:start + batch_size])
            if not torch.isfinite(losses["total"]):
                raise ValueError("Nonfinite training loss")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            # Report unweighted validation action MSE, never task success.
            sums, frames = {}, {}
            for name, partition in (("train", indices), ("validation", validation)):
                total, count = 0., 0
                for start in range(0, len(partition), batch_size):
                    index = partition[start:start + batch_size]
                    n = int(tensors["action_mask"][index].sum())
                    total += float(loss_for(index)["action"]) * n
                    count += n
                sums[name], frames[name] = total / count, count
            history.append(dict(epoch=epoch + 1, train_action_mse=sums["train"], validation_action_mse=sums["validation"]))
    output.mkdir(parents=True, exist_ok=False)
    run_metadata = dict(cache_sha256=hashlib.sha256(cache_bytes).hexdigest(),
        split_sha256=hashlib.sha256(split_bytes).hexdigest(), metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(), seed=seed,
        train_episodes=split["train"], validation_episodes=split["validation"],
        train_examples=len(indices), validation_examples=len(validation), cache_metadata=metadata,
        torch_version=str(torch.__version__), config=asdict(config), epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, note="Offline adapter training; cache provenance is declared by its producer, not verified by model inference.")
    save_adapter(output / "adapter.pt", model, run_metadata)
    write_json(output / "run.json", run_metadata)
    write_json(output / "losses.json", history)
    return dict(output_dir=str(output.resolve()), train_examples=len(indices), validation_examples=len(validation),
                final_validation_action_mse=history[-1]["validation_action_mse"], scope=metadata.get("source", "user_cache"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "split", "metadata", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mode", choices=("history", "failure_aware"), default="failure_aware")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(train(args.cache, args.split, args.metadata, args.output_dir, mode=args.mode, epochs=args.epochs, seed=args.seed)))
        return 0
    except (ValueError, OSError, KeyError, ImportError) as exc:
        parser.exit(2, f"Adapter training failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
