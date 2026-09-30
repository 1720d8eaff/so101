"""Validate, select and split portable episode manifests; fit train-only statistics."""
import argparse
import random
from pathlib import Path
import numpy as np
from .io import read_json, read_jsonl, unique_ids, write_json, write_jsonl


def load_episode(manifest_path, row):
    filename = row.get("npz_path")
    if not isinstance(filename, str) or not filename:
        raise ValueError("npz_path must be a non-empty path")
    path = Path(filename)
    if not path.is_absolute():
        path = Path(manifest_path).resolve().parent / path
    with np.load(path, allow_pickle=False) as episode:
        if not {"observation_state", "action"} <= set(episode.files):
            raise ValueError("NPZ must contain observation_state and action")
        state = episode["observation_state"].copy()
        action = episode["action"].copy()
    if state.ndim != 2 or action.ndim != 2 or state.shape[0] != action.shape[0] or not state.shape[0] or not state.shape[1] or not action.shape[1]:
        raise ValueError("Expected aligned, non-empty [T, D] state/action arrays")
    if state.dtype.kind not in "iuf" or action.dtype.kind not in "iuf":
        raise ValueError("State/action arrays must be real-valued numeric arrays")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("State/action contain NaN or infinity")
    return state.astype(np.float64), action.astype(np.float64)


def validate_manifest(path):
    rows = read_jsonl(path)
    unique_ids(rows)
    if not rows:
        raise ValueError("Manifest is empty")
    expected_dimensions = None
    results = []
    for row in rows:
        item = {"episode_id": row["episode_id"], "valid": False}
        try:
            state, action = load_episode(path, row)
            dimensions = (state.shape[1], action.shape[1])
            if expected_dimensions is None:
                expected_dimensions = dimensions
            if dimensions != expected_dimensions:
                raise ValueError("Inconsistent state/action dimensions across episodes")
            for name in ("success", "timeout", "collision", "abnormal_action"):
                if name in row and not isinstance(row[name], bool):
                    raise ValueError(f"{name} must be a boolean")
            item.update(valid=True, frames=state.shape[0], state_dim=state.shape[1], action_dim=action.shape[1])
        except (ValueError, OSError, TypeError) as exc:
            item["error"] = str(exc)
        results.append(item)
    return rows, results


def select_rows(rows, checks, require_success=True):
    valid = {str(x["episode_id"]): x["valid"] for x in checks}
    selected, rejected = [], []
    for row in rows:
        reasons = []
        if not valid[str(row["episode_id"])]:
            reasons.append("invalid_arrays")
        if require_success and row.get("success") is not True:
            reasons.append("success_not_confirmed")
        for name in ("timeout", "collision", "abnormal_action"):
            if row.get(name) is not False:
                reasons.append(f"{name}_true_or_unknown")
        if reasons:
            rejected.append({"episode_id": row["episode_id"], "reasons": reasons})
        else:
            selected.append(row)
    return selected, rejected


def split_episodes(rows, train_fraction, seed):
    ids = sorted(unique_ids(rows))
    if not 0 < train_fraction < 1 or len(ids) < 2:
        raise ValueError("Need at least 2 episodes and 0 < train_fraction < 1")
    random.Random(seed).shuffle(ids)
    count = max(1, min(len(ids) - 1, int(len(ids) * train_fraction)))
    return {"seed": seed, "train": sorted(ids[:count]), "validation": sorted(ids[count:])}


def fit_statistics(path, rows, split):
    known = unique_ids(rows)
    train = [str(x) for x in split["train"]]
    validation = [str(x) for x in split["validation"]]
    if not train or len(set(train)) != len(train) or len(set(validation)) != len(validation):
        raise ValueError("Split lists must be unique and training must be non-empty")
    if set(train) & set(validation) or set(train + validation) != known:
        raise ValueError("Split must cover the manifest exactly without train/validation overlap")
    # Merge per-episode moments in float64; memory scales with one trajectory.
    totals = {}
    for row in rows:
        if str(row["episode_id"]) not in set(train):
            continue
        state, action = load_episode(path, row)
        for key, array in (("observation_state", state), ("action", action)):
            n, mean, m2 = array.shape[0], array.mean(axis=0), ((array - array.mean(axis=0)) ** 2).sum(axis=0)
            if key not in totals:
                totals[key] = [n, mean, m2, array.min(axis=0), array.max(axis=0)]
            else:
                old_n, old_mean, old_m2, minimum, maximum = totals[key]
                if mean.shape != old_mean.shape:
                    raise ValueError("Inconsistent channel dimensions")
                delta, total = mean - old_mean, old_n + n
                totals[key] = [total, old_mean + delta * n / total, old_m2 + m2 + delta ** 2 * old_n * n / total,
                               np.minimum(minimum, array.min(axis=0)), np.maximum(maximum, array.max(axis=0))]
    output = {"training_episode_ids": train, "format": "portable_npz_stats_v1", "statistics": {}}
    for key, (n, mean, m2, minimum, maximum) in totals.items():
        output["statistics"][key] = {"frames": n, "mean": mean.tolist(), "std": np.sqrt(np.maximum(m2 / n, 0)).tolist(), "min": minimum.tolist(), "max": maximum.tolist()}
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("validate", "filter", "split", "stats"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", required=True)
        command.add_argument("--output", required=True)
        if name == "filter":
            command.add_argument("--keep-unsuccessful", action="store_true")
            command.add_argument("--rejected", required=True)
        if name == "split":
            command.add_argument("--train-fraction", type=float, default=0.8)
            command.add_argument("--seed", type=int, default=0)
        if name == "stats":
            command.add_argument("--split", required=True)
    args = parser.parse_args(argv)
    try:
        if args.action in ("validate", "filter"):
            rows, checks = validate_manifest(args.manifest)
            if args.action == "validate":
                write_json(args.output, {"episodes": checks})
                return 0 if all(x["valid"] for x in checks) else 1
            selected, rejected = select_rows(rows, checks, not args.keep_unsuccessful)
            # Preserve paths when the filtered manifest lives in a different folder.
            for row in selected:
                p = Path(row["npz_path"])
                if not p.is_absolute():
                    row["npz_path"] = str((Path(args.manifest).resolve().parent / p).resolve())
            write_jsonl(args.output, selected)
            write_json(args.rejected, rejected)
            return 0 if selected else 1
        rows = read_jsonl(args.manifest)
        if args.action == "split":
            write_json(args.output, split_episodes(rows, args.train_fraction, args.seed))
        else:
            write_json(args.output, fit_statistics(args.manifest, rows, read_json(args.split)))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
