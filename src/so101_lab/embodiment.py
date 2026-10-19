"""Body-frame EEF delta + absolute gripper, with explicit FK/IK contracts.

No Panda/UR5/SO-101 kinematics are fabricated here: callers supply a calibrated
FK model and joint limits. This module implements the shared representation
and a bounded numerical IK backend, independently of a specific simulator.
"""
import numpy as np
from .runtime import vector


def check_pose(value):
    t = np.asarray(value, dtype=np.float64)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
        or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8)
        or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(t[:3, :3]), 1., atol=1e-6)):
        raise ValueError("Expected a proper finite SE(3) matrix")
    return t


def exp_rotation(value):
    v = vector(value, "rotation vector")
    if len(v) != 3:
        raise ValueError("Rotation vector needs three radians")
    x, y, z = v
    k = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = np.linalg.norm(v)
    if angle < 1e-7:
        return np.eye(3) + k + .5 * k @ k
    return np.eye(3) + np.sin(angle) / angle * k + (1 - np.cos(angle)) / angle ** 2 * k @ k


def log_rotation(rotation):
    r = np.asarray(rotation, dtype=float)
    angle = np.arccos(np.clip((np.trace(r) - 1) / 2, -1., 1.))
    vee = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if angle < 1e-7:
        return .5 * vee
    if np.pi - angle < 1e-5:
        values, axes = np.linalg.eigh((r + r.T) / 2)
        axis = axes[:, np.argmax(values)]
        if np.dot(axis, vee) < 0:
            axis *= -1
        return axis * angle
    return vee * angle / (2 * np.sin(angle))


def encode_delta(current, target, gripper):
    current, target = check_pose(current), check_pose(target)
    if isinstance(gripper, bool) or not np.isfinite(gripper) or not 0 <= gripper <= 1:
        raise ValueError("Gripper is absolute normalized opening in [0,1]")
    rotation = current[:3, :3].T
    delta_position = rotation @ (target[:3, 3] - current[:3, 3])
    delta_rotation = log_rotation(rotation @ target[:3, :3])
    return np.r_[delta_position, delta_rotation, gripper]


def decode_delta(current, action):
    current = check_pose(current)
    action = vector(action, "unified action")
    if action.shape != (7,) or not 0 <= action[-1] <= 1:
        raise ValueError("Action must be [body dx,dy,dz,rotvec(3),gripper]")
    delta = np.eye(4)
    delta[:3, :3], delta[:3, 3] = exp_rotation(action[3:6]), action[:3]
    return current @ delta, float(action[-1])


class IKConvergenceError(ValueError):
    pass


class KinematicsAdapter:
    def __init__(self, forward_kinematics, lower, upper, *, rotation_scale_m=.15,
                 max_iterations=100, position_tolerance_m=1e-4, rotation_tolerance_rad=1e-3):
        self.fk = forward_kinematics
        self.lower, self.upper = vector(lower, "lower"), vector(upper, "upper")
        if (self.lower.shape != self.upper.shape or np.any(self.lower >= self.upper)
            or type(max_iterations) is not int or max_iterations < 1
            or any(not np.isfinite(v) or v <= 0 for v in
                   (rotation_scale_m, position_tolerance_m, rotation_tolerance_rad))):
            raise ValueError("Invalid IK bounds or tolerances")
        self.rotation_scale, self.max_iterations = rotation_scale_m, max_iterations
        self.position_tolerance, self.rotation_tolerance = position_tolerance_m, rotation_tolerance_rad

    def joints(self, value):
        q = vector(value, "joints")
        if q.shape != self.lower.shape or np.any(q < self.lower) or np.any(q > self.upper):
            raise ValueError("Joints outside declared limits")
        return q

    def encode(self, current_joints, target_joints, gripper):
        return encode_delta(self.fk(self.joints(current_joints)), self.fk(self.joints(target_joints)), gripper)

    def decode(self, current_joints, action):
        seed = self.joints(current_joints)
        target, gripper = decode_delta(self.fk(seed), action)
        return self.solve(target, seed), gripper

    def solve(self, target, seed):
        target, q = check_pose(target), self.joints(seed)
        for _ in range(self.max_iterations):
            current = check_pose(self.fk(q))
            dp = target[:3, 3] - current[:3, 3]
            dr = log_rotation(target[:3, :3] @ current[:3, :3].T)
            if np.linalg.norm(dp) <= self.position_tolerance and np.linalg.norm(dr) <= self.rotation_tolerance:
                return q
            jacobian = np.empty((6, len(q)))
            for j in range(len(q)):
                trial = q.copy()
                # One-sided difference points into the joint interval at a limit.
                epsilon = min(1e-5, (self.upper[j] - self.lower[j]) / 100)
                if q[j] + epsilon > self.upper[j]:
                    epsilon *= -1
                trial[j] += epsilon
                perturbed = check_pose(self.fk(trial))
                jacobian[:3, j] = (perturbed[:3, 3] - current[:3, 3]) / epsilon
                jacobian[3:, j] = self.rotation_scale * log_rotation(perturbed[:3, :3] @ current[:3, :3].T) / epsilon
            error = np.r_[dp, self.rotation_scale * dr]
            dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-6 * np.eye(6), error)
            dq *= min(1., .1 / max(np.max(np.abs(dq)), 1e-12))
            q = np.clip(q + dq, self.lower, self.upper)
        raise IKConvergenceError("Target is unreachable or IK did not converge; no command returned")


def robot_task_holdout(episodes, heldout_pairs):
    """Split entire episodes by (robot_id, task); never split frames randomly."""
    heldout = set(heldout_pairs)
    if not heldout:
        raise ValueError("Need explicit held-out robot/task combinations")
    train, validation, seen, found = [], [], set(), set()
    for row in episodes:
        identifier = str(row["episode_id"])
        if identifier in seen or not row["robot_id"] or not row["task"]:
            raise ValueError("Duplicate episode or missing robot/task")
        seen.add(identifier)
        pair = (row["robot_id"], row["task"])
        if pair in heldout:
            validation.append(identifier)
            found.add(pair)
        else:
            train.append(identifier)
    if found != heldout or not train or not validation:
        raise ValueError("Unknown holdout pair or empty split")
    return dict(train=sorted(train), validation=sorted(validation), heldout_pairs=sorted(heldout))
