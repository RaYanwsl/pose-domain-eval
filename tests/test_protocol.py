from __future__ import annotations

import numpy as np

from pose_eval.align import similarity_align
from pose_eval.evaluate import evaluate_samples
from pose_eval.metrics import frame_metrics


def test_similarity_align_removes_a_global_scale():
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(24, 3))
    gt[0] = 0
    pred = gt * 1.8 + np.array([0.2, -0.1, 0.4])
    aligned = similarity_align(pred, gt)
    assert np.allclose(aligned, gt, atol=1e-6)


def test_root_mpjpe_keeps_scale_while_pa_removes_it():
    gt = np.zeros((24, 3))
    gt[:, 0] = np.linspace(-0.4, 0.4, 24)
    gt[:, 2] = np.linspace(0.2, 1.6, 24)
    pred = gt * 1.5
    metrics = frame_metrics(pred, gt)
    assert metrics["mpjpe"] > 40
    assert metrics["pa_mpjpe"] < 1


def test_domain_report_separates_scale_limbs_and_time():
    result = evaluate_samples()
    domains = {item["domain"]: item for item in result["by_domain"]}
    source = domains["source"]
    outdoor = domains["outdoor"]
    occlusion = domains["occlusion"]
    assert source["mpjpe"] < 30
    assert outdoor["mpjpe"] > outdoor["pa_mpjpe"] + 15
    assert occlusion["arm_pa"] > occlusion["leg_pa"] + 20
    assert outdoor["foot_skate"] > source["foot_skate"] + 5
    assert outdoor["heading_jump"] > source["heading_jump"] + 2
    text = "\n".join(result["conclusion"])
    assert "尺度" in text
    assert "上肢" in text
    assert "脚滑" in text
