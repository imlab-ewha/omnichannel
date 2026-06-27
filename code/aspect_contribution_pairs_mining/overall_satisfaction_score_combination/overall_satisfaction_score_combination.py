"""
overall_satisfaction_score_combination.py

Computes a review-level overall satisfaction score by combining the star rating
and the sentiment probability derived from comment text (see Eq. 1 in the paper).

    R      = rating - 1          (scales 1–5 → 0–4)
    Rmax   = 4

    Overall Satisfaction = R              if sentiment == negative
                         = Rmax + p_pos   if sentiment == positive

The non-overlapping ranges [0, Rmax] and [Rmax, Rmax+1] reflect the asymmetric
reliability of ratings across sentiment polarities.

Usage:
    python overall_satisfaction_score_combination.py [--input PATH] [--output-dir DIR]
"""

import argparse
from pathlib import Path

import pandas as pd

_SCRIPT_DIR         = Path(__file__).resolve().parent
_RESOURCE_DIR       = _SCRIPT_DIR.parents[2] / "resource"
_DEFAULT_INPUT      = _RESOURCE_DIR / "4_sentiment_analysis" / "sentiment_analysis.csv"
_DEFAULT_OUTPUT_DIR = _RESOURCE_DIR / "4_sentiment_analysis"

_RMAX = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine rating and sentiment probability into an overall satisfaction score.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help="Input CSV with 'rating', 'sentiment', and 'positivity_probability' columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory where the output CSV will be written.",
    )
    return parser.parse_args()


def compute_overall_satisfaction(df: pd.DataFrame) -> pd.Series:
    R = df["rating"] - 1
    return R.where(df["sentiment"] == "negative", _RMAX + df["positivity_probability"])


def run(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input.resolve())

    df["overall_satisfaction"] = compute_overall_satisfaction(df)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "overall_satisfaction.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Overall satisfaction score computed for {len(df)} reviews.")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    run(parse_args())
