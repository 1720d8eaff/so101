"""Real named-joint MuJoCo FK backend for the existing numerical IK adapter.

Callers supply their own licensed/calibrated MJCF or URDF and end-effector body
or site. No Panda, UR5e or SO-101 geometry is synthesized by this module.
"""
import hashlib
from pathlib import Path
import numpy as np
from .embodiment import KinematicsAdapter


class MujocoKinematics:
    def __init__(self, model_path, joint_names, *, eef_site=None, eef_body=None):
        import mujoco
        if bool(eef_site) == bool(eef_body) or not joint_names or len(set(joint_names)) != len(joint_names):
            raise ValueError("Provide unique joints and exactly one end-effector site or body")
        self.mujoco = mujoco
        path = Path(model_path).resolve(strict=True)
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        ids = [self.model.joint(n).id for n in joint_names]
        kinds = [self.model.jnt_type[j] for j in ids]
        if any(t not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) for t in kinds):
            raise ValueError("Only scalar hinge/slide joints are supported")
        if not all(self.model.jnt_limited[j] for j in ids):
            raise ValueError("Each controlled joint needs explicit finite limits")
        self.q_indices = np.array([self.model.jnt_qposadr[j] for j in ids])
        self.lower, self.upper = self.model.jnt_range[ids].T.copy()
        self.site = self.model.site(eef_site).id if eef_site else None
        self.body = self.model.body(eef_body).id if eef_body else None
        self.provenance = dict(model_path=str(path), model_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               joint_names=list(joint_names), eef_site=eef_site, eef_body=eef_body,
                               note="Hash covers the main XML/URDF only; record all included assets separately")

    def __call__(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != self.lower.shape or not np.isfinite(q).all() or np.any(q < self.lower) or np.any(q > self.upper):
            raise ValueError("FK input outside model joint limits")
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self.q_indices] = q
        self.mujoco.mj_forward(self.model, self.data)
        pose = np.eye(4)
        if self.site is not None:
            pose[:3, 3], pose[:3, :3] = self.data.site_xpos[self.site], self.data.site_xmat[self.site].reshape(3, 3)
        else:
            pose[:3, 3], pose[:3, :3] = self.data.xpos[self.body], self.data.xmat[self.body].reshape(3, 3)
        return pose

    def adapter(self, **kwargs):
        return KinematicsAdapter(self, self.lower, self.upper, **kwargs)
