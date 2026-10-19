"""Reproducible FK/IK checks on pinned upstream models; no task success claims."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .mujoco_backend import MujocoKinematics
from .embodiment import log_rotation

MODELS={
    "panda":dict(file="franka_emika_panda/panda.xml",joints=[f"joint{i}" for i in range(1,8)],body="hand"),
    "ur5e":dict(file="universal_robots_ur5e/ur5e.xml",joints=["shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"],site="attachment_site")}
PIN="62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35"


def run(root,samples=32,seed=20261019):
    root=Path(root);rng=np.random.default_rng(seed);results={}
    for name,c in MODELS.items():
        backend=MujocoKinematics(root/c["file"],c["joints"],eef_site=c.get("site"),eef_body=c.get("body"))
        home=backend.model.key_qpos[0,backend.q_indices].copy()
        adapter=backend.adapter(max_iterations=200)
        errors=[]
        for _ in range(samples):
            current=np.clip(home+rng.normal(0,.12,len(home)),backend.lower+.01,backend.upper-.01)
            target=np.clip(current+rng.normal(0,.035,len(home)),backend.lower+.01,backend.upper-.01)
            action=adapter.encode(current,target,float(rng.uniform(.1,.9)))
            solved,gripper=adapter.decode(current,action)
            actual,expected=backend(solved),backend(target)
            errors.append(dict(position_m=float(np.linalg.norm(actual[:3,3]-expected[:3,3])),
                rotation_rad=float(np.linalg.norm(log_rotation(actual[:3,:3]@expected[:3,:3].T))),
                gripper=float(gripper)))
        if max(r["position_m"] for r in errors)>adapter.position_tolerance or max(r["rotation_rad"] for r in errors)>adapter.rotation_tolerance:
            raise AssertionError("FK/IK tolerance failed")
        model_dir=(root/c["file"]).parent
        assets={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(model_dir.rglob("*")) if p.is_file()}
        results[name]=dict(samples=samples,checks=errors,asset_sha256=assets,
            joint_names=c["joints"],end_effector=c.get("site") or c.get("body"))
    return dict(passed=True,source="MuJoCo_model_kinematics",real_robot=False,seed=seed,
        upstream="https://github.com/google-deepmind/mujoco_menagerie",expected_upstream_commit=PIN,
        scope="Local reachable pose roundtrips only; no cross-robot policy or task evaluation",models=results)


def main():
    p=argparse.ArgumentParser();p.add_argument("--menagerie",required=True);p.add_argument("--output",required=True)
    args=p.parse_args();result=run(args.menagerie);path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!="models"},indent=2))


if __name__=="__main__":main()
