"""Markdown and JSON reports."""

from __future__ import annotations

import json
from pathlib import Path

from pose_eval.evaluate import DOMAIN_LABELS


def write_report(out_dir: Path, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(_markdown(result), encoding="utf-8")


def _markdown(result: dict) -> str:
    lines = [
        "# 三维人体姿态跨域泛化评测",
        "",
        "源域和目标域使用同一套 24 关节、同一根节点对齐和同一种相似对齐。单位是毫米。",
        "",
        "| 域 | 样本 | MPJPE | PA-MPJPE | 上肢 PA | 下肢 PA | 脚滑 mm/帧 | 朝向跳变 ° |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in result["by_domain"]:
        lines.append(
            f"| {item['label']} | {item['count']} | {item['mpjpe']:.1f} | {item['pa_mpjpe']:.1f} | "
            f"{item['arm_pa']:.1f} | {item['leg_pa']:.1f} | {item['foot_skate']:.1f} | {item['heading_jump']:.1f} |"
        )
    lines.extend(["", "## 结论", ""])
    lines.extend(f"- {line}" for line in result["conclusion"])
    lines.extend(["", "## 逐样本", "", "| 样本 | 域 | 划分 | MPJPE | PA-MPJPE | 脚滑 | 朝向跳变 |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"])
    for row in result["samples"]:
        lines.append(
            f"| {row['sample_id']} | {DOMAIN_LABELS.get(row['domain'], row['domain'])} | {row['split']} | "
            f"{row['mpjpe']:.1f} | {row['pa_mpjpe']:.1f} | {row['foot_skate']:.1f} | {row['heading_jump']:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)
