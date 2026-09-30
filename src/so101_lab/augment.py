"""Create offline action-label variants using explicitly selected channels."""
import argparse
from pathlib import Path
import numpy as np
from .io import read_json, write_json


def variants(actions, channels, std, scale, alpha, seed):
    actions = np.asarray(actions, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if actions.ndim != 2 or not actions.shape[0] or std.shape != (actions.shape[1],):
        raise ValueError("Expected non-empty [T,D] actions and D training standard deviations")
    if not np.isfinite(actions).all() or not np.isfinite(std).all() or (std < 0).any():
        raise ValueError("Actions and standard deviations must be finite, and std >= 0")
    if not channels or len(channels) != len(set(channels)) or any(c < 0 or c >= actions.shape[1] for c in channels):
        raise ValueError("Select unique, in-range action channel indices")
    if not np.isfinite(scale) or scale < 0 or not 0 < alpha <= 1:
        raise ValueError("Require finite scale >= 0 and 0 < alpha <= 1")
    noisy = actions.copy()
    noisy[:, channels] += np.random.default_rng(seed).normal(size=(len(actions), len(channels))) * std[channels] * scale
    filtered = noisy.copy()
    for t in range(1, len(actions)):
        filtered[t, channels] = alpha * noisy[t, channels] + (1 - alpha) * filtered[t - 1, channels]
    return noisy, filtered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="One portable NPZ episode")
    parser.add_argument("--train-stats", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--channels", type=int, nargs="+", required=True, help="Noise/filter these indices only; explicitly exclude gripper channels")
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        out = Path(args.output_dir)
        if out.exists():
            raise ValueError("output-dir already exists; choose a new directory")
        with np.load(args.input, allow_pickle=False) as episode:
            values = {key: episode[key].copy() for key in episode.files}
        if "action" not in values:
            raise ValueError("NPZ must contain action")
        if values["action"].dtype.kind not in "iuf":
            raise ValueError("Actions must be real numeric values")
        stats = read_json(args.train_stats)
        noisy, filtered = variants(values["action"], args.channels, stats["statistics"]["action"]["std"], args.scale, args.alpha, args.seed)
        out.mkdir(parents=True)
        for name, action in (("clean", values["action"]), ("noisy", noisy), ("filtered", filtered)):
            np.savez_compressed(out / f"{name}.npz", **{**values, "action": action})
        write_json(out / "variant-config.json", {"source": str(Path(args.input).resolve()), "training_stats": str(Path(args.train_stats).resolve()),
                   "channels": args.channels, "scale": args.scale, "alpha": args.alpha, "seed": args.seed,
                   "note": "Offline labels only; observations unchanged. No joint-limit clipping is performed."})
        return 0
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
