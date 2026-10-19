"""Reproducible CPU smoke demo on explicitly synthetic data, with no robot access."""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import numpy as np
from .io import write_json
from .recovery import action_targets, history_windows
from .runtime import ActionChunk, BufferSettings, StateAlignedBuffer


def synthetic_cache(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    parts = []
    for episode in range(4):
        t = np.linspace(0., 1., 16, dtype=np.float32)
        states = np.column_stack((t, np.sin(t * 3 + episode * .1), np.full_like(t, episode * .1)))
        actions = np.column_stack((.05 * states[:, 0], -.04 * states[:, 1]))
        history, history_mask = history_windows(states, 4)
        targets, action_mask = action_targets(actions, 4)
        labels = np.zeros(len(t), dtype=np.int64)
        labels[7:10] = 2  # Synthetic grasp failure supervision only.
        parts.append(dict(history_state=history, history_mask=history_mask, base_actions=np.zeros_like(targets),
            target_actions=targets, action_mask=action_mask, failure_target=labels, progress_target=t,
            episode_id=np.full(len(t), f"ep-{episode}")))
    arrays = {key: np.concatenate([p[key] for p in parts], axis=0) for key in parts[0]}
    np.savez_compressed(directory / "cache.npz", **arrays)
    split = dict(train=["ep-0", "ep-1", "ep-2"], validation=["ep-3"])
    base_spec = b'{"policy":"synthetic_zero_chunk","horizon":4,"action_dim":2}'
    (directory / "synthetic-base.json").write_bytes(base_spec)
    write_json(directory / "split.json", split)
    write_json(directory / "metadata.json", dict(source="synthetic_cpu_demo_not_robot_experiment",
        base_checkpoint_sha256=hashlib.sha256(base_spec).hexdigest(), normalization_training_episodes=split["train"],
        action_space="upstream_processed_absolute_joint_targets", preprocessing="Synthetic identity preprocessing; not a pretrained SmolVLA cache."))
    return directory / "cache.npz", directory / "split.json", directory / "metadata.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists():
            raise ValueError("Choose a fresh demo output directory")
        # Import before creating files, so a missing optional extra leaves no partial run.
        from .train_adapter import train
        import torch
        args.output_dir.mkdir(parents=True, exist_ok=False)
        paths = synthetic_cache(args.output_dir / "synthetic-data")
        training = train(*paths, args.output_dir / "adapter-run", epochs=20)
        buffer = StateAlignedBuffer([0., 0.], [-1., -1.], [1., 1.], .05,
            BufferSettings(max_pending_steps=16, replan_below_steps=3))
        result = buffer.submit(ActionChunk(0, 0., .08, 1 / 30, [0., 0.], np.ones((12, 2)) * .3), .08, [0., 0.])
        trace = []
        for i in range(3, 17):
            step = buffer.tick(i / 30, i / 30)
            trace.append(dict(t=i / 30, action=step.action.tolist(), status=step.status,
                              request_replan=step.request_replan, pending=step.pending_steps))
        write_json(args.output_dir / "demo.json", dict(scope="synthetic smoke test; virtual scheduling time; no task success or physical latency measured",
            training=training, chunk_acceptance=result, counters=dict(buffer.counters), control_trace=trace))
        print(f"Synthetic offline demo passed: {(args.output_dir / 'demo.json').resolve()}")
        return 0
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Offline demo failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
