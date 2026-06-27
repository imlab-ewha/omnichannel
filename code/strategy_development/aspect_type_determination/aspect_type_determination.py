"""
aspect_type_determination.py

Classifies each normalized aspect as "search" or "experience" based on
whether it can be evaluated before or only after product use.
Uses the OpenAI Chat API with one call per aspect.

Inputs:
  --aspects   resource/3_aspect_selection/top_aspects.csv

Outputs:
  resource/6_aspect_type/aspect_types.csv

Usage:
    python aspect_type_determination.py [--aspects PATH] [--output PATH]
                                        [--model MODEL] [--temperature T]
                                        [--seed N]
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_SCRIPT_DIR         = Path(__file__).resolve().parent
_RESOURCE_DIR       = _SCRIPT_DIR.parents[2] / "resource"
_DEFAULT_ASPECTS    = _RESOURCE_DIR / "3_aspect_selection" / "top_aspects.csv"
_DEFAULT_OUTPUT     = _RESOURCE_DIR / "6_aspect_type" / "aspect_types.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
Role description: You are an expert in determining product aspects extracted from customer reviews of beauty and health products, including the following categories: skincare, face masks, cleansers, dermocosmetics, hair care, perfume & diffuser, supplements, and oral care.

Task description: Your task is to determine the given product aspect into either "search" or "experience" based on the following rules.

Determination rules:
If the quality of an aspect can be assessed prior to using the product, it is determined as a search aspect; otherwise, it is determined as an experience aspect.

Task output: Return only a JSON object.
{
"aspect": "aspect",
"type": "search" or "experience",
"reason": "one sentence explanation in Korean"
}
"""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Determine aspects as 'search' or 'experience' via the OpenAI Chat API.",
    )
    parser.add_argument(
        "--aspects",
        type=Path,
        default=_DEFAULT_ASPECTS,
        help="CSV file with a 'normalized_aspect' column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI chat model identifier.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible outputs.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_aspects(path: Path) -> list[str]:
    return pd.read_csv(path)["normalized_aspect"].dropna().tolist()


def determine_aspect(
    client: OpenAI,
    aspect: str,
    *,
    model: str,
    temperature: float,
    seed: int,
) -> dict:
    response = client.chat.completions.create(
        model=model,
        seed=seed,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Aspect: {aspect}"},
        ],
    )
    result = json.loads(response.choices[0].message.content)
    return {
        "aspect": aspect,
        "type":   result.get("type", "unknown"),
        "reason": result.get("reason", ""),
    }


def run(args: argparse.Namespace) -> None:
    aspects = load_aspects(args.aspects.resolve())
    client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    results = []

    for aspect in aspects:
        result = determine_aspect(
            client, aspect,
            model=args.model,
            temperature=args.temperature,
            seed=args.seed,
        )
        results.append(result)

    df = pd.DataFrame(results)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Determination complete. {len(df)} aspects processed.")
    print(df["type"].value_counts().to_string())
    print(f"\nSaved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run(parse_args())
