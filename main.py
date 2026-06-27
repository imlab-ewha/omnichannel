"""
main.py

End-to-end pipeline: example_review.csv → scenario_assignment.csv

Usage:
    python main.py [--input PATH] [--resource-dir DIR]

Arguments:
    --input         Path to input review CSV
                    (default: resource/example_review.csv)
    --resource-dir  Base directory for all intermediate and output files
                    (default: resource/)

Pipeline steps:
  1. Aspect extraction         → resource/1_aspect_extraction/
  2. Aspect normalization      → resource/2_aspect_normalization/
  3. Aspect selection          → resource/3_aspect_selection/
  4. Sentiment analysis        → resource/4_sentiment_analysis/
  5. Overall satisfaction      → resource/4_sentiment_analysis/
  6. Model training            → resource/5_aspect_contribution/
  7. SHAP analysis             → resource/5_aspect_contribution/
  8. Aspect type determination → resource/6_aspect_type/
  9. Scenario assignment       → resource/7_scenario/
"""

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_CODE_DIR   = _SCRIPT_DIR / "code"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end omnichannel pipeline.")
    parser.add_argument(
        "--input",
        type=Path,
        default=_SCRIPT_DIR / "resource" / "example_review.csv",
        help="Input review CSV (default: resource/example_review.csv).",
    )
    parser.add_argument(
        "--resource-dir",
        type=Path,
        default=_SCRIPT_DIR / "resource",
        help="Base directory for intermediate and output files (default: resource/).",
    )
    return parser.parse_args()


def build_steps(input_csv: Path, r: Path) -> list[tuple[str, Path, str, list[str]]]:
    return [
        (
            "Step 1 — Aspect extraction",
            _CODE_DIR / "frequent_aspect_mining" / "aspect_extraction",
            "aspect_extraction.py",
            ["--input", str(input_csv),
             "--output", str(r / "1_aspect_extraction")],
        ),
        (
            "Step 2 — Aspect normalization",
            _CODE_DIR / "frequent_aspect_mining" / "aspect_normalization",
            "aspect_normalization.py",
            ["--input", str(r / "1_aspect_extraction" / "aspect_extraction.csv"),
             "--output-dir", str(r / "2_aspect_normalization")],
        ),
        (
            "Step 3 — Aspect selection",
            _CODE_DIR / "frequent_aspect_mining" / "aspect_selection",
            "aspect_selection.py",
            ["--input", str(r / "2_aspect_normalization"),
             "--output-dir", str(r / "3_aspect_selection")],
        ),
        (
            "Step 4 — Sentiment analysis",
            _CODE_DIR / "aspect_contribution_pairs_mining" / "overall_satisfaction_score_combination",
            "sentiment_analysis.py",
            ["--input", str(input_csv),
             "--output-dir", str(r / "4_sentiment_analysis")],
        ),
        (
            "Step 5 — Overall satisfaction score",
            _CODE_DIR / "aspect_contribution_pairs_mining" / "overall_satisfaction_score_combination",
            "overall_satisfaction_score_combination.py",
            ["--input", str(r / "4_sentiment_analysis" / "sentiment_analysis.csv"),
             "--output-dir", str(r / "4_sentiment_analysis")],
        ),
        (
            "Step 6 — Model training",
            _CODE_DIR / "aspect_contribution_pairs_mining" / "aspect_contribution_calculation",
            "model_training.py",
            ["--satisfaction", str(r / "4_sentiment_analysis" / "overall_satisfaction.csv"),
             "--top-aspects",  str(r / "3_aspect_selection" / "top_aspects.csv"),
             "--reviews",      str(r / "3_aspect_selection" / "top_aspect_reviews.csv"),
             "--output-dir",   str(r / "5_aspect_contribution")],
        ),
        (
            "Step 7 — SHAP analysis",
            _CODE_DIR / "aspect_contribution_pairs_mining" / "aspect_contribution_calculation",
            "aspect_contribution_calculation.py",
            ["--model-dir",  str(r / "5_aspect_contribution" / "models"),
             "--output-dir", str(r / "5_aspect_contribution")],
        ),
        (
            "Step 8 — Aspect type determination",
            _CODE_DIR / "strategy_development" / "aspect_type_determination",
            "aspect_type_determination.py",
            ["--aspects", str(r / "3_aspect_selection" / "top_aspects.csv"),
             "--output",  str(r / "6_aspect_type" / "aspect_types.csv")],
        ),
        (
            "Step 9 — Scenario assignment",
            _CODE_DIR / "strategy_development" / "aspect_scenario_assignment",
            "aspect_scenario_assignment.py",
            ["--shap",       str(r / "5_aspect_contribution" / "shap.csv"),
             "--types",      str(r / "6_aspect_type" / "aspect_types.csv"),
             "--output-dir", str(r / "7_scenario")],
        ),
    ]


def run_step(label: str, script_dir: Path, script: str, extra_args: list[str]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    subprocess.run([sys.executable, script] + extra_args, cwd=script_dir, check=True)


if __name__ == "__main__":
    args = parse_args()
    steps = build_steps(args.input, args.resource_dir)

    for label, script_dir, script, extra_args in steps:
        run_step(label, script_dir, script, extra_args)

    output = args.resource_dir / "7_scenario" / "scenario_assignment.csv"
    print(f"\nPipeline complete.")
    print(f"Final output: {output}")
