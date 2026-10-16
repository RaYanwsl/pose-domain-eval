"""Root alignment and similarity Procrustes."""

from __future__ import annotations

import numpy as np


def root_align(joints: np.ndarray, root: int = 0) -> np.ndarray:
    return joints - joints[root]


def similarity_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Map pred onto gt with translation, rotation, and scale."""

    if pred.shape != gt.shape or pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError(f"关节数组必须是 (J, 3)，收到 pred {pred.shape}、gt {gt.shape}")
    mu_pred = pred.mean(axis=0)
    mu_gt = gt.mean(axis=0)
    centered_pred = pred - mu_pred
    centered_gt = gt - mu_gt
    covariance = centered_pred.T @ centered_gt / pred.shape[0]
    left, singular, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_t) < 0:
        correction[-1, -1] = -1
    rotation = left @ correction @ right_t
    variance = float((centered_pred**2).sum() / pred.shape[0])
    if variance <= 1e-12:
        raise ValueError("预测关节退化成一个点，无法做相似对齐")
    scale = float(np.trace(np.diag(singular) @ correction) / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"相似对齐尺度无效: {scale}")
    return scale * centered_pred @ rotation + mu_gt
