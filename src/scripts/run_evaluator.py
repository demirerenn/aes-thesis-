"""Evaluator Agent CLI runner.

Thin wrapper around src.agents.nodes.evaluator.evaluate_run that replaces the
long inline python -c block in HANDOFF §4.4.

Usage:
    python -m src.scripts.run_evaluator \
        --run-dir runs/asap1-p2-deberta_v3_lg-corn-pilot-rev0 \
        --config configs/baseline_asap1_p2.yaml

Defaults assume the Jinja2 template lives at src/agents/templates/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.nodes.evaluator import EvaluatorInput, evaluate_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluator Agent — render run report + gating decision")
    ap.add_argument("--run-dir", required=True, help="Training run directory containing summary_*.json")
    ap.add_argument("--config", required=True, help="Config YAML used for the run")
    ap.add_argument("--template-dir", default="src/agents/templates",
                    help="Jinja2 template directory (default: src/agents/templates)")
    args = ap.parse_args()

    result = evaluate_run(EvaluatorInput(
        run_dir=Path(args.run_dir),
        config_path=Path(args.config),
        template_dir=Path(args.template_dir),
    ))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
