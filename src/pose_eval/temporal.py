"""Foot skating and root-heading jumps. Z is up, units are meters."""

from __future__ import annotations

import numpy as np

from pose_eval.skeleton import FOOT_JOINTS, NECK, PELVIS


def _angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom < 1e-8:
        return 0.0
    cosine = float(np.clip(np.dot(left, right) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def foot_skate_mm(joints: np.ndarray, height_m: float = 0.04) -> float:
    """Mean horizontal travel of feet that stay near the ground."""

    if joints.shape[0] < 2:
        return 0.0
    feet = joints[:, FOOT_JOINTS, :]
    previous = feet[:-1]
    current = feet[1:]
    planted = (previous[..., 2] <= height_m) & (current[..., 2] <= height_m)
    if not np.any(planted):
        return 0.0
    travel = np.linalg.norm(current[..., :2] - previous[..., :2], axis=-1)
    return float(travel[planted].mean() * 1000.0)


def heading_jump_deg(joints: np.ndarray) -> float:
    """Mean frame-to-frame change of the pelvis-to-neck heading on the ground plane."""

    if joints.shape[0] < 2:
        return 0.0
    jumps = []
    previous = None
    for frame in joints:
        heading = frame[NECK, :2] - frame[PELVIS, :2]
        if previous is not None:
            jumps.append(_angle_deg(previous, heading))
        if np.linalg.norm(heading) >= 1e-8:
            previous = heading
    if not jumps:
        return 0.0
    return float(np.mean(jumps))
