"""MPJPE and PA-MPJPE in millimeters, plus per-group error."""

from __future__ import annotations

import numpy as np

from pose_eval.align import root_align, similarity_align
from pose_eval.skeleton import ARM_JOINTS, HEAD_JOINTS, LEG_JOINTS, TORSO_JOINTS


def _mpjpe_mm(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.linalg.norm(pred - gt, axis=1).mean() * 1000.0)


def frame_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    rooted_pred = root_align(pred)
    rooted_gt = root_align(gt)
    aligned = similarity_align(pred, gt)
    per_joint = np.linalg.norm(aligned - gt, axis=1) * 1000.0
    return {
        "mpjpe": _mpjpe_mm(rooted_pred, rooted_gt),
        "pa_mpjpe": _mpjpe_mm(aligned, gt),
        "arm_pa": float(per_joint[list(ARM_JOINTS)].mean()),
        "leg_pa": float(per_joint[list(LEG_JOINTS)].mean()),
        "head_pa": float(per_joint[list(HEAD_JOINTS)].mean()),
        "torso_pa": float(per_joint[list(TORSO_JOINTS)].mean()),
    }


def sequence_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    if pred.shape != gt.shape or pred.ndim != 3:
        raise ValueError("序列必须是对齐后的 (T, J, 3)")
    rooted_pred = pred - pred[:, :1, :]
    rooted_gt = gt - gt[:, :1, :]
    mpjpe = float(np.linalg.norm(rooted_pred - rooted_gt, axis=2).mean() * 1000.0)
    aligned = np.empty_like(pred, dtype=np.float64)
    for index in range(pred.shape[0]):
        aligned[index] = similarity_align(pred[index], gt[index])
    per_joint = np.linalg.norm(aligned - gt, axis=2) * 1000.0
    return {
        "mpjpe": mpjpe,
        "pa_mpjpe": float(per_joint.mean()),
        "arm_pa": float(per_joint[:, list(ARM_JOINTS)].mean()),
        "leg_pa": float(per_joint[:, list(LEG_JOINTS)].mean()),
        "head_pa": float(per_joint[:, list(HEAD_JOINTS)].mean()),
        "torso_pa": float(per_joint[:, list(TORSO_JOINTS)].mean()),
    }
