"""
aspect_extraction.py

Extracts Aspect-Opinion-Sentiment (AOS) triplets from Korean product reviews
using a fine-tuned KcELECTRA-based model.

Usage:
    python aspect_extraction.py [--input PATH] [--output PATH]
                                [--batch-size N] [--gpu-id N]

The input CSV must contain a 'preprocessed_content' column.
"""

import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import argparse
import json
import logging
import logging.config
from pathlib import Path

import pandas as pd

_SCRIPT_DIR         = Path(__file__).resolve().parent
_LOG_DIR            = _SCRIPT_DIR / "resources" / "log"
_RESOURCE_DIR       = _SCRIPT_DIR.parents[2] / "resource"
_DEFAULT_INPUT      = _RESOURCE_DIR / "example_review.csv"
_DEFAULT_OUTPUT_DIR = _RESOURCE_DIR / "1_aspect_extraction"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_log_config = json.loads((_SCRIPT_DIR / "logger.json").read_text())
_log_config["handlers"]["file_debug"]["filename"] = str(_LOG_DIR / "debug.log")
_log_config["handlers"]["file_error"]["filename"] = str(_LOG_DIR / "error.log")
logging.config.dictConfig(_log_config)

from processor.review import ReviewProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract AOS triplets from Korean product reviews.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help="Input CSV file containing a 'preprocessed_content' column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Output directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of sentences per model forward pass.",
    )
    parser.add_argument(
        "--gpu-id",
        type=str,
        default="0",
        help="CUDA device index (CUDA_VISIBLE_DEVICES).",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input.resolve())
    df = df.rename(columns={"preprocessed_content": "corrected_text"})
    records = df.to_dict(orient="records")

    processor = ReviewProcessor(
        device="cuda",
        batch_size=args.batch_size,
        gpu_id=args.gpu_id,
    )
    results = processor.tagging(records)

    output_path = args.output.resolve()
    if output_path.suffix != ".csv":
        output_path = output_path / "aspect_extraction.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame(results)[
        ["sid", "preprocessed_content", "sentence", "raw_aspect", "aspect_pos", "aspect"]
    ]
    result_df = result_df.drop_duplicates(subset=["sid", "raw_aspect"]).reset_index(drop=True)
    result_df.to_csv(output_path, encoding="utf-8-sig", index=False)

    print(f"Extraction complete. {len(result_df)} AOS triplets from {len(df)} reviews.")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    run(parse_args())
