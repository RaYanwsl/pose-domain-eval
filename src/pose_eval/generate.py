"""Deterministic source and target poses.

These are protocol fixtures, not THuman scans and not a trained pose model.
The error pattern is fixed so a laptop can show which domain shift moves
which number.
"""

from __future__ import annotations

import numpy as np

from pose_eval.skeleton import ARM_JOINTS, HEAD_JOINTS, JOINT_NAMES, TORSO_JOINTS

HEIGHT = len(JOINT_NAMES)


def rest_pose() -> np.ndarray:
    pose = np.zeros((HEIGHT, 3), dtype=np.float64)
    pose[0] = (0.00, 0.00, 0.95)
    pose[1] = (-0.10, 0.00, 0.90)
    pose[2] = (0.10, 0.00, 0.90)
    pose[3] = (0.00, 0.00, 1.08)
    pose[4] = (-0.10, 0.02, 0.48)
    pose[5] = (0.10, 0.02, 0.48)
    pose[6] = (0.00, 0.00, 1.22)
    pose[7] = (-0.11, 0.04, 0.08)
    pose[8] = (0.11, 0.04, 0.08)
    pose[9] = (0.00, 0.00, 1.36)
    pose[10] = (-0.12, 0.10, 0.02)
    pose[11] = (0.12, 0.10, 0.02)
    pose[12] = (0.00, 0.00, 1.50)
    pose[13] = (-0.08, 0.00, 1.40)
    pose[14] = (0.08, 0.00, 1.40)
    pose[15] = (0.00, 0.00, 1.68)
    pose[16] = (-0.18, 0.00, 1.42)
    pose[17] = (0.18, 0.00, 1.42)
    pose[18] = (-0.42, 0.00, 1.40)
    pose[19] = (0.42, 0.00, 1.40)
    pose[20] = (-0.64, 0.00, 1.38)
    pose[21] = (0.64, 0.00, 1.38)
    pose[22] = (-0.72, 0.00, 1.36)
    pose[23] = (0.72, 0.00, 1.36)
    return pose


def _walk(
    frames: int,
    skate: float,
    yaw_step_deg: float,
    noise: float,
    seed: int,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    gt = np.repeat(rest_pose()[None], frames, axis=0)
    left_contact = None
    right_contact = None
    for index in range(frames):
        gt[index, :, 1] += index * 0.04
        phase = index % 8
        left_down = phase < 4
        if left_down:
            gt[index, 7, 2] = 0.02
            gt[index, 10, 2] = 0.015
            gt[index, 8, 2] = 0.12
            gt[index, 11, 2] = 0.11
        else:
            gt[index, 8, 2] = 0.02
            gt[index, 11, 2] = 0.015
            gt[index, 7, 2] = 0.12
            gt[index, 10, 2] = 0.11
        yaw = np.radians(yaw_step_deg if index % 5 == 0 else 0.0)
        if yaw:
            rotation = np.array(
                [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
            )
            pelvis = gt[index, 0].copy()
            gt[index] = (gt[index] - pelvis) @ rotation.T + pelvis
        if left_down:
            if phase == 0 or left_contact is None:
                left_contact = gt[index, [7, 10], :2].copy()
            gt[index, 7, :2] = left_contact[0]
            gt[index, 10, :2] = left_contact[1]
            right_contact = None
        else:
            if phase == 4 or right_contact is None:
                right_contact = gt[index, [8, 11], :2].copy()
            gt[index, 8, :2] = right_contact[0]
            gt[index, 11, :2] = right_contact[1]
            left_contact = None
    pred = gt.copy()
    pred += rng.normal(0, noise, size=pred.shape)
    # A constant offset does not move between frames. Accumulate so planted feet slide.
    pred[:, FOOT_INDEXES, 1] += skate * np.arange(frames)[:, None]
    if scale != 1.0:
        pelvis = pred[:, 0:1, :].copy()
        pred = (pred - pelvis) * scale + pelvis
    return gt, pred


FOOT_INDEXES = np.array([7, 8, 10, 11])


def _single(noise: float, seed: int, scale: float = 1.0, arm_bias: float = 0.0, torso_bias: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    gt = rest_pose()[None]
    pred = gt.copy()
    pred += rng.normal(0, noise, size=pred.shape)
    pelvis = pred[0, 0].copy()
    pred[0] = (pred[0] - pelvis) * scale + pelvis
    pred[0, list(ARM_JOINTS), 0] += arm_bias
    pred[0, list(TORSO_JOINTS), 0] += torso_bias
    pred[0, list(HEAD_JOINTS), 2] += torso_bias * 0.2
    return gt, pred


def build_samples() -> list[dict]:
    samples = [
        _pack("synth_walk", "source", "source", *_walk(16, skate=0.0, yaw_step_deg=1.0, noise=0.004, seed=1)),
        _pack("synth_stand", "source", "source", *_single(0.005, seed=2)),
        _pack("indoor_room", "indoor", "target", *_single(0.012, seed=3)),
        _pack("indoor_hall", "indoor", "target", *_walk(16, skate=0.004, yaw_step_deg=2.0, noise=0.008, seed=4)),
        _pack("outdoor_street", "outdoor", "target", *_walk(16, skate=0.0, yaw_step_deg=1.0, noise=0.008, seed=5, scale=1.45)),
        _pack("outdoor_plaza", "outdoor", "target", *_walk(16, skate=0.035, yaw_step_deg=18.0, noise=0.01, seed=6)),
        _pack("occlusion_arms", "occlusion", "target", *_single(0.008, seed=7, arm_bias=0.14)),
        _pack("occlusion_bag", "occlusion", "target", *_single(0.008, seed=8, arm_bias=0.11)),
        _pack("clothing_coat", "clothing", "target", *_single(0.01, seed=9, torso_bias=0.045)),
        _pack("clothing_skirt", "clothing", "target", *_single(0.01, seed=10, torso_bias=0.035)),
        _pack("far_10m", "far", "target", *_single(0.045, seed=11)),
        _pack("far_16m", "far", "target", *_single(0.055, seed=12)),
    ]
    return samples


def _pack(sample_id: str, domain: str, split: str, gt: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "sample_id": sample_id,
        "domain": domain,
        "split": split,
        "gt": gt.astype(np.float32),
        "pred": pred.astype(np.float32),
    }
