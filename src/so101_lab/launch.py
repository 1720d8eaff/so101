"""Build LeRobot 0.6.1 commands; preview unless --execute is supplied."""
import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path
from .io import read_json

LEROBOT_VERSION = "0.6.1"
ACTIONS = ("calibrate-follower", "calibrate-leader", "teleoperate", "record", "train-smolvla", "train-act", "train-diffusion", "evaluate-robot")


def flag(name, value):
    if isinstance(value, (dict, list, bool)):
        value = json.dumps(value, separators=(",", ":"))
    return f"--{name}={value}"


def build_command(action, cfg):
    robot = cfg["robot"]
    leader = cfg["teleop"]
    dataset = cfg["dataset"]
    training = cfg["training"]
    evaluation = cfg["evaluation"]
    follower_flags = [flag("robot.type", "so101_follower"), flag("robot.port", robot["port"]), flag("robot.id", robot["id"])]
    leader_flags = [flag("teleop.type", "so101_leader"), flag("teleop.port", leader["port"]), flag("teleop.id", leader["id"])]
    if action == "calibrate-follower":
        return ["lerobot-calibrate", *follower_flags]
    if action == "calibrate-leader":
        return ["lerobot-calibrate", *leader_flags]
    cameras = flag("robot.cameras", robot["cameras"])
    if action == "teleoperate":
        return ["lerobot-teleoperate", *follower_flags, cameras, *leader_flags]
    if action in ("record", "evaluate-robot"):
        section = dataset if action == "record" else evaluation
        executable = "lerobot-record" if action == "record" else "lerobot-rollout"
        cmd = [executable, *follower_flags, cameras,
               flag("dataset.repo_id", section["repo_id"]), flag("dataset.root", section["root"]),
               flag("dataset.single_task", section["task"]), flag("dataset.fps", section["fps"]),
               flag("dataset.num_episodes", section["num_episodes"]),
               flag("dataset.episode_time_s", section["episode_time_s"]),
               flag("dataset.reset_time_s", section["reset_time_s"]),
               flag("dataset.push_to_hub", False), flag("dataset.no_stamp", True), flag("display_data", False)]
        if action == "record":
            cmd.extend(leader_flags)
        else:
            cmd.extend([flag("strategy.type", "episodic"), flag("inference.type", "sync"),
                        flag("policy.path", evaluation["policy_path"]), flag("device", training["device"]),
                        flag("fps", evaluation["fps"]), flag("rename_map", evaluation.get("rename_map", {}))])
        return cmd
    policy = action.removeprefix("train-")
    cmd = ["lerobot-train", flag("dataset.repo_id", dataset["repo_id"]), flag("dataset.root", dataset["root"]),
           flag("policy.device", training["device"]), flag("policy.push_to_hub", False),
           flag("output_dir", str(Path(training["output_root"]) / policy)),
           flag("job_name", f"so101_{policy}"), flag("steps", training["steps"]),
           flag("batch_size", training["batch_size"]), flag("seed", training["seed"]), flag("wandb.enable", False)]
    if policy == "smolvla":
        cmd.append(flag("policy.path", training["smolvla_base"]))
        cmd.append(flag("rename_map", training.get("rename_map", {})))
    elif training.get("init_policies", {}).get(policy):
        cmd.append(flag("policy.path", training["init_policies"][policy]))
    else:
        cmd.append(flag("policy.type", policy))
    if training.get("episodes") is not None:
        episodes = training["episodes"]
        if not episodes or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in episodes) or len(set(episodes)) != len(episodes):
            raise ValueError("training.episodes must contain unique non-negative integer LeRobot episode indices")
        cmd.append(flag("dataset.episodes", episodes))
    return cmd


def doctor():
    print(f"Python: {sys.version.split()[0]}")
    try:
        version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        version = "not installed"
    print(f"LeRobot: {version}; expected {LEROBOT_VERSION}")
    print(f"Training / hardware Python requirement: >=3.12; satisfied: {sys.version_info >= (3, 12)}")
    for name in ("lerobot-train", "lerobot-record", "lerobot-rollout", "lerobot-calibrate", "lerobot-teleoperate"):
        print(f"{name}: {shutil.which(name) or 'not installed'}")
    return 0 if version == LEROBOT_VERSION and sys.version_info >= (3, 12) else 1


def validate_execution(action, cfg):
    try:
        version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError("Install the robot extra in a Python 3.12 environment first") from exc
    if version != LEROBOT_VERSION:
        raise ValueError(f"Expected LeRobot {LEROBOT_VERSION}, found {version}")
    if action in ("record", "evaluate-robot"):
        section = cfg["dataset"] if action == "record" else cfg["evaluation"]
        if Path(section["root"]).exists():
            raise ValueError("Recording root already exists; choose a fresh root and repo_id for this run")
        if action == "evaluate-robot":
            policy = Path(section["policy_path"])
            if not (policy / "config.json").is_file():
                raise ValueError("evaluation.policy_path must point to a local pretrained_model directory containing config.json")
    if action.startswith("train-"):
        if not (Path(cfg["dataset"]["root"]) / "meta" / "info.json").is_file():
            raise ValueError("dataset.root must contain a local LeRobot dataset, including meta/info.json")
        output = Path(cfg["training"]["output_root"]) / action.removeprefix("train-")
        if output.exists():
            raise ValueError("Training output already exists; change training.output_root to preserve the previous run")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("doctor", *ACTIONS))
    parser.add_argument("--config", default="configs/so101.example.json")
    parser.add_argument("--execute", action="store_true", help="Run the printed command; otherwise preview only")
    args = parser.parse_args(argv)
    if args.action == "doctor":
        return doctor()
    try:
        cfg = read_json(args.config)
        cmd = build_command(args.action, cfg)
        # A JSON argv list is unambiguous across PowerShell and POSIX shells.
        print(json.dumps({"action": args.action, "argv": cmd}, ensure_ascii=False, indent=2))
        if not args.execute:
            return 0
        validate_execution(args.action, cfg)
        if not shutil.which(cmd[0]):
            raise ValueError(f"Executable not found: {cmd[0]}")
        return subprocess.run(cmd, check=False).returncode
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
