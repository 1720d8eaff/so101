"""Audit declared collection, paired trial and measured latency records.

These checks establish consistency of supplied records, not robot provenance
or historical authenticity. No samples or experimental outcomes are generated.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from .io import read_jsonl, unique_ids, write_json
from .report import summarize


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing {name}")
    return value


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected lowercase SHA256")
    return value


def source_check(rows, source):
    if source not in {"real_robot", "simulation", "synthetic"} or not rows:
        raise ValueError("Need nonempty records and explicit source")
    if any(row.get("source") != source for row in rows):
        raise ValueError("Mixed/unknown data sources; simulation frames cannot count as real episodes")


def timestamp(value):
    stamp = datetime.fromisoformat(text(value, "timestamp").replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Collection timestamps must include a timezone")
    return stamp


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def collection(rows, source, root, expected_total=None, expected_accepted=None):
    source_check(rows, source)
    unique_ids(rows)
    root, verified, total_frames, accepted, videos = Path(root).resolve(), {}, 0, 0, 0
    for row in rows:
        for key in ("robot", "task", "operator_role"):
            text(row.get(key), key)
        if timestamp(row["ended_at"]) <= timestamp(row["started_at"]):
            raise ValueError("Episode end must follow start")
        total_frames += integer(row["frames"], "frames", 1)
        if type(row.get("accepted")) is not bool:
            raise ValueError("accepted must be a reviewed boolean")
        if not row["accepted"]:
            text(row.get("exclusion_reason"), "exclusion_reason")
        accepted += row["accepted"]
        artifacts = row["artifacts"]
        if not isinstance(artifacts, dict) or not {"trajectory", "calibration"} <= set(artifacts):
            raise ValueError("Every episode needs trajectory and calibration references")
        videos += "video" in artifacts
        for role, artifact in artifacts.items():
            relative = Path(text(artifact.get("path"), "artifact path"))
            path = (root / relative).resolve()
            if relative.is_absolute() or root not in path.parents:
                raise ValueError("Artifact paths must stay inside the manifest directory")
            expected = digest(artifact["sha256"])
            if path not in verified:
                verified[path] = file_hash(path)
            if verified[path] != expected:
                raise ValueError(f"Artifact digest mismatch: {role}/{relative}")
    for expected, actual, name in ((expected_total, len(rows), "collected episodes"), (expected_accepted, accepted, "accepted episodes")):
        if expected is not None and expected != actual:
            raise ValueError(f"Claim mismatch for {name}: expected {expected}, found {actual}")
    return dict(source=source, collected_episodes=len(rows), accepted_episodes=accepted,
        rejected_episodes=len(rows) - accepted, frames=total_frames, episodes_with_video=videos,
        unique_verified_artifacts=len(verified), historical_authenticity_proven=False)


def trials(rows, source):
    source_check(rows, source)
    summary = summarize(rows)
    policies = sorted({row["policy"] for row in rows})
    if len(policies) < 2:
        raise ValueError("Paired evidence needs at least two policies")
    pairs, checkpoints = defaultdict(dict), {}
    for row in rows:
        checkpoint = digest(row["checkpoint_sha256"])
        model_key = (row["policy"], str(row["seed"]))
        model_value = (row["checkpoint"], checkpoint)
        if model_key in checkpoints and checkpoints[model_key] != model_value:
            raise ValueError("Checkpoint changed within one policy/training seed comparison")
        checkpoints[model_key] = model_value
        protocol, initial = digest(row["protocol_sha256"]), digest(row["initial_condition_sha256"])
        scene, repeat = text(row["scene_id"], "scene_id"), integer(row["repeat"], "repeat")
        environment_seed = integer(row["environment_seed"], "environment_seed")
        key = (str(row["seed"]), row["task"], row["condition"], scene, repeat)
        if row["policy"] in pairs[key]:
            raise ValueError("Duplicate policy within a paired scene/repeat")
        pairs[key][row["policy"]] = (protocol, initial, environment_seed)
    for group in pairs.values():
        if set(group) != set(policies) or len(set(group.values())) != 1:
            raise ValueError("Unpaired trials or mismatched protocol/initial condition/environment seed")
    recovery = []
    if any("recognized_failure" in row or "autonomous_recovery" in row for row in rows):
        for row in rows:
            if type(row.get("recognized_failure")) is not bool or type(row.get("autonomous_recovery")) is not bool:
                raise ValueError("All recovery annotations must be explicit booleans")
            if row["autonomous_recovery"] and not row["recognized_failure"]:
                raise ValueError("Recovery without a recognized failure")
        for group in summary["groups"]:
            fields = ("policy", "checkpoint", "task", "condition", "seed")
            selected = [r for r in rows if all(str(r[k]) == str(group[k]) for k in fields) and r["recognized_failure"]]
            recovered = sum(r["autonomous_recovery"] for r in selected)
            recovery.append(dict(**{k: group[k] for k in fields}, recognized_failures=len(selected), autonomous_recoveries=recovered,
                                 recovery_rate=recovered / len(selected) if selected else None))
    return {**summary, "source": source, "paired_conditions": len(pairs), "trial_count": len(rows), "recovery": recovery,
            "note": "Checkpoint/protocol hashes and outcomes are supplied annotations; no hardware or model execution is attested."}


def distribution(values):
    values = sorted(values)
    def percentile(fraction):
        position = (len(values) - 1) * fraction
        lo = int(position)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (position - lo)
    return dict(samples=len(values), mean=statistics.fmean(values), p50=percentile(.5), p95=percentile(.95), max=max(values))


def latency(rows, source, target_hz=30.):
    source_check(rows, source)
    if not math.isfinite(target_hz) or target_hz <= 0:
        raise ValueError("target_hz must be positive")
    groups, seen, previous, clocks, episode_times = defaultdict(list), set(), {}, {}, {}
    stages = ("capture_ns", "preprocess_end_ns", "inference_start_ns", "inference_end_ns", "send_ns", "execute_ns")
    for row in rows:
        mode, trial = text(row["mode"], "mode"), text(row["trial_id"], "trial_id")
        step = integer(row["step"], "step")
        key, trial_key = (mode, trial, step), (mode, trial)
        if key in seen:
            raise ValueError("Duplicate control tick")
        seen.add(key)
        clock = text(row["clock_id"], "clock_id")
        if trial_key in clocks and clocks[trial_key] != clock:
            raise ValueError("Clock domains changed within a trial")
        clocks[trial_key] = clock
        start, end, blocked = [integer(row[k], k) for k in ("control_start_ns", "control_end_ns", "blocked_ns")]
        if (end < start or blocked > end - start or (trial_key in previous and start <= previous[trial_key])):
            raise ValueError("Unordered tick or impossible blocked interval")
        previous[trial_key] = start
        times = [integer(row[k], k) for k in stages]
        if times != sorted(times):
            raise ValueError("Pipeline stages reordered or clocks unsynchronized")
        if "episode_start_ns" in row or "episode_end_ns" in row:
            a, b = integer(row["episode_start_ns"], "episode_start_ns"), integer(row["episode_end_ns"], "episode_end_ns")
            if b <= a or a > start or end > b or (trial_key in episode_times and episode_times[trial_key] != (a, b)):
                raise ValueError("Inconsistent episode boundaries")
            episode_times[trial_key] = (a, b)
        groups[mode].append(dict(blocked_ms=blocked / 1e6, control_ms=(end - start) / 1e6,
            action_age_ms=(times[-1] - times[0]) / 1e6, inference_ms=(times[3] - times[2]) / 1e6,
            deadline_miss=end - start > 1e9 / target_hz))
    output = []
    for mode, selected in sorted(groups.items()):
        duration = [(b - a) / 1e9 for (m, trial), (a, b) in episode_times.items() if m == mode]
        output.append(dict(mode=mode, ticks=len(selected), target_hz=target_hz,
            **{key: distribution([r[key] for r in selected]) for key in ("blocked_ms", "control_ms", "action_age_ms", "inference_ms")},
            deadline_misses=sum(r["deadline_miss"] for r in selected), episode_duration_s=distribution(duration) if duration else None))
    return dict(source=source, groups=output,
        note="Blocking is per control tick; action age is capture-to-execute. Declared clock synchronization and physical execution are not attested.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("collection", "trials", "latency"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", required=True, choices=("real_robot", "simulation", "synthetic"))
    parser.add_argument("--expected-total", type=int)
    parser.add_argument("--expected-accepted", type=int)
    parser.add_argument("--target-hz", type=float, default=30.)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("Refusing to overwrite evidence report")
        rows = read_jsonl(args.input)
        if args.kind == "collection":
            result = collection(rows, args.source, args.input.parent, args.expected_total, args.expected_accepted)
        elif args.kind == "trials":
            result = trials(rows, args.source)
        else:
            result = latency(rows, args.source, args.target_hz)
        write_json(args.output, dict(input_sha256=file_hash(args.input), **result))
        print(f"Record consistency checks passed: {args.output.resolve()}")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"Evidence check failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
