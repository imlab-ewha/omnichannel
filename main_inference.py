"""
main_inference.py

End-to-end pipeline: data/example_review.csv → outputs/scenario_assignment/scenario_assignment.csv

Usage:
    python main.py [--input PATH] [--output-dir DIR]
"""

import argparse
import logging
from pathlib import Path

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

import src.aspect_extraction                 as aspect_extraction
import src.aspect_normalization              as aspect_normalization
import src.aspect_selection                  as aspect_selection
import src.sentiment_analysis                as sentiment_analysis
import src.overall_satisfaction_combination  as overall_satisfaction_combination
import src.regressor_training                as regressor_training
import src.contribution_calculation          as contribution_calculation
import src.type_determination                as type_determination
import src.scenario_assignment               as scenario_assignment

_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end omnichannel pipeline.")
    parser.add_argument("--input",      type=Path, default=_ROOT / "data" / "example_review.csv")
    parser.add_argument("--output-dir", type=Path, default=_ROOT / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    r    = args.output_dir.resolve()

    print("\n" + "=" * 60)
    print("  Step 1 — Aspect extraction")
    print("=" * 60)
    args1 = aspect_extraction.parse_args([])
    args1.input  = args.input
    args1.output = r / "aspect_extraction"
    aspect_extraction.run(args1)

    print("\n" + "=" * 60)
    print("  Step 2 — Aspect normalization")
    print("=" * 60)
    args2 = aspect_normalization.parse_args([])
    args2.input      = r / "aspect_extraction" / "aspect_extraction.csv"
    args2.output_dir = r / "aspect_normalization"
    aspect_normalization.run(args2)

    print("\n" + "=" * 60)
    print("  Step 3 — Aspect selection")
    print("=" * 60)
    args3 = aspect_selection.parse_args([])
    args3.input      = r / "aspect_normalization"
    args3.output_dir = r / "aspect_selection"
    aspect_selection.run(args3)

    print("\n" + "=" * 60)
    print("  Step 4 — Sentiment analysis")
    print("=" * 60)
    args4 = sentiment_analysis.parse_args([])
    args4.input      = args.input
    args4.output_dir = r / "sentiment_analysis"
    sentiment_analysis.run(args4)

    print("\n" + "=" * 60)
    print("  Step 5 — Overall satisfaction combination")
    print("=" * 60)
    args5 = overall_satisfaction_combination.parse_args([])
    args5.input      = r / "sentiment_analysis" / "sentiment_analysis.csv"
    args5.output_dir = r / "overall_satisfaction_combination"
    overall_satisfaction_combination.run(args5)

    print("\n" + "=" * 60)
    print("  Step 6 — Regressor training")
    print("=" * 60)
    args6 = regressor_training.parse_args([])
    args6.satisfaction = r / "overall_satisfaction_combination" / "overall_satisfaction.csv"
    args6.top_aspects  = r / "aspect_selection" / "top_aspects.csv"
    args6.reviews      = r / "aspect_selection" / "top_aspect_reviews.csv"
    args6.output_dir   = r / "regressor_training"
    regressor_training.run(args6)

    print("\n" + "=" * 60)
    print("  Step 7 — Contribution calculation")
    print("=" * 60)
    args7 = contribution_calculation.parse_args([])
    args7.model_dir  = r / "regressor_training" / "models"
    args7.output_dir = r / "contribution_calculation"
    contribution_calculation.run(args7)

    print("\n" + "=" * 60)
    print("  Step 8 — Type determination")
    print("=" * 60)
    args8 = type_determination.parse_args([])
    args8.aspects = r / "aspect_selection" / "top_aspects.csv"
    args8.output  = r / "type_determination" / "aspect_types.csv"
    type_determination.run(args8)

    print("\n" + "=" * 60)
    print("  Step 9 — Scenario assignment")
    print("=" * 60)
    args9 = scenario_assignment.parse_args([])
    args9.shap       = r / "contribution_calculation" / "shap.csv"
    args9.types      = r / "type_determination" / "aspect_types.csv"
    args9.output_dir = r / "scenario_assignment"
    scenario_assignment.run(args9)

    print(f"\nPipeline complete.")
    print(f"Final output: {r / 'scenario_assignment' / 'scenario_assignment.csv'}")


if __name__ == "__main__":
    main()
