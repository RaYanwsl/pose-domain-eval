"""Score every fixture with the same joint definition and alignment."""

from __future__ import annotations

from pose_eval.generate import build_samples
from pose_eval.metrics import sequence_metrics
from pose_eval.temporal import foot_skate_mm, heading_jump_deg


DOMAIN_ORDER = ("source", "indoor", "outdoor", "occlusion", "clothing", "far")
DOMAIN_LABELS = {
    "source": "合成源域",
    "indoor": "室内",
    "outdoor": "室外",
    "occlusion": "遮挡",
    "clothing": "着装",
    "far": "远距离",
}


def evaluate_samples() -> dict:
    rows = []
    for sample in build_samples():
        metrics = sequence_metrics(sample["pred"], sample["gt"])
        metrics["foot_skate"] = foot_skate_mm(sample["pred"])
        metrics["heading_jump"] = heading_jump_deg(sample["pred"])
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "domain": sample["domain"],
                "split": sample["split"],
                **metrics,
            }
        )
    return {"samples": rows, "by_domain": _by_domain(rows), "conclusion": _conclusion(rows)}


def _by_domain(rows: list[dict]) -> list[dict]:
    grouped = []
    present = {row["domain"] for row in rows}
    ordered = [domain for domain in DOMAIN_ORDER if domain in present]
    ordered.extend(sorted(present - set(ordered)))
    keys = ("mpjpe", "pa_mpjpe", "arm_pa", "leg_pa", "foot_skate", "heading_jump")
    for domain in ordered:
        members = [row for row in rows if row["domain"] == domain]
        summary = {"domain": domain, "label": DOMAIN_LABELS.get(domain, domain), "count": len(members)}
        for key in keys:
            summary[key] = float(sum(row[key] for row in members) / len(members))
        grouped.append(summary)
    return grouped


def _conclusion(rows: list[dict]) -> list[str]:
    domains = {item["domain"]: item for item in _by_domain(rows)}
    source = domains["source"]
    target_rows = [row for row in rows if row["split"] == "target"]
    target_mpjpe = sum(row["mpjpe"] for row in target_rows) / len(target_rows)
    target_pa = sum(row["pa_mpjpe"] for row in target_rows) / len(target_rows)
    lines = [
        (
            f"合成源域 MPJPE {source['mpjpe']:.1f} mm，PA-MPJPE {source['pa_mpjpe']:.1f} mm。"
            f"同一关节定义和根节点对齐下，真实目标域 MPJPE {target_mpjpe:.1f} mm，"
            f"PA-MPJPE {target_pa:.1f} mm。"
        )
    ]
    outdoor = domains["outdoor"]
    if outdoor["mpjpe"] > outdoor["pa_mpjpe"] + 15:
        lines.append(
            f"室外 MPJPE {outdoor['mpjpe']:.1f} mm，PA-MPJPE {outdoor['pa_mpjpe']:.1f} mm。"
            "尺度偏移主要留在未做相似对齐的 MPJPE 里。"
        )
    occlusion = domains["occlusion"]
    if occlusion["arm_pa"] > occlusion["leg_pa"] + 20:
        lines.append(
            f"遮挡样本的上肢 PA 误差 {occlusion['arm_pa']:.1f} mm，下肢 {occlusion['leg_pa']:.1f} mm。"
            "掉点集中在被挡住的四肢，不是全身均匀变差。"
        )
    if outdoor["foot_skate"] > source["foot_skate"] + 5:
        lines.append(
            f"脚滑：源域 {source['foot_skate']:.1f} mm/帧，室外 {outdoor['foot_skate']:.1f} mm/帧。"
            f"朝向跳变：源域 {source['heading_jump']:.1f}°，室外 {outdoor['heading_jump']:.1f}°。"
            "单帧均值看不到这两项。"
        )
    worst = max(
        (item for item in domains.values() if item["domain"] != "source"),
        key=lambda item: item["mpjpe"],
    )
    lines.append(f"目标域里 MPJPE 最高的是{worst['label']}（{worst['mpjpe']:.1f} mm）。")
    return lines
