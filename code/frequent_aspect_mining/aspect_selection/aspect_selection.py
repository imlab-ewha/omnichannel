"""
aspect_selection.py

Selects the top-k most frequent normalized aspects from a normalization result CSV
and saves two output files:
  - top_aspects.csv       : ranked list of top-k aspects with occurrence counts
  - top_aspect_reviews.csv: review rows whose normalized_aspect is in the top-k set

Usage:
    python aspect_selection.py [--input PATH] [--output-dir DIR] [--top-k N]

If --input is a directory, the most recently modified normalization_*.csv
inside that directory is used automatically.
"""

import argparse
from pathlib import Path

import pandas as pd

_SCRIPT_DIR         = Path(__file__).resolve().parent
_RESOURCE_DIR       = _SCRIPT_DIR.parents[2] / "resource"
_DEFAULT_INPUT      = _RESOURCE_DIR / "2_aspect_normalization"
_DEFAULT_OUTPUT_DIR = _RESOURCE_DIR / "3_aspect_selection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select top-k normalized aspects by frequency.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help=(
            "Normalization result CSV, or the directory containing it "
            "If a directory is given, the latest normalization_*.csv is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory where output files will be written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of top aspects to select.",
    )
    return parser.parse_args()


def resolve_input(path: Path) -> Path:
    if path.is_dir():
        candidates = sorted(path.glob("normalization_*.csv"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No normalization_*.csv found in {path}")
        return candidates[-1]
    return path


def run(args: argparse.Namespace) -> None:
    input_path = resolve_input(args.input.resolve())
    df = pd.read_csv(input_path)

    valid = df[df["normalized_aspect"].notna() & (df["normalized_aspect"] != "n/a")].copy()

    counts = valid["normalized_aspect"].value_counts()
    top_aspects = counts.head(args.top_k).reset_index()
    top_aspects.columns = ["normalized_aspect", "count"]

    top_set = set(top_aspects["normalized_aspect"])
    top_reviews = valid[valid["normalized_aspect"].isin(top_set)].reset_index(drop=True)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    top_aspects.to_csv(output_dir / "top_aspects.csv", index=False, encoding="utf-8-sig")
    top_reviews.to_csv(output_dir / "top_aspect_reviews.csv", index=False, encoding="utf-8-sig")

    print(f"Selected top {len(top_aspects)} aspects from {counts.shape[0]} unique normalized aspects.")
    print(f"Saved: {output_dir / 'top_aspects.csv'}")
    print(f"Saved: {output_dir / 'top_aspect_reviews.csv'}")


if __name__ == "__main__":
    run(parse_args())
