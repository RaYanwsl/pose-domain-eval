"""Command line."""

from __future__ import annotations

import argparse
from pathlib import Path

from pose_eval.evaluate import evaluate_samples
from pose_eval.report import write_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pose-eval", description="三维人体姿态跨域泛化评测")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate_samples()
    write_report(args.out, result)
    print("\n".join(result["conclusion"]))
    print(f"report: {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
