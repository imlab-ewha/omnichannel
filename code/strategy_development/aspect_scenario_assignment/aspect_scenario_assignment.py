"""
assignment.py

Assigns an omnichannel strategy scenario to each aspect based on:
  - Aspect type  (search / experience) from resource/6_aspect_type/aspect_types.csv
  - Dominant channel (online / offline) from resource/5_aspect_contribution/shap_results.csv

Scenario matrix:
  S1 -- Search     x Online dominant
  S2 -- Search     x Offline dominant
  S3 -- Experience x Online dominant
  S4 -- Experience x Offline dominant

Dominance: the channel with the higher mean SHAP value is dominant.
Aspects where |online_mean_shap - offline_mean_shap| < epsilon are marked N/A.

Inputs:
  --shap        resource/5_aspect_contribution/shap_results.csv
  --types       resource/6_aspect_type/aspect_types.csv
  --output-dir  resource/7_scenario/
  --epsilon     minimum |online - offline| to assign a scenario (default: 0.001)

Output:
  resource/7_scenario/scenario_assignment.csv  (columns: aspect, scenario)
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

_SCRIPT_DIR         = Path(__file__).resolve().parent
_RESOURCE_DIR       = _SCRIPT_DIR.parents[2] / "resource"
_DEFAULT_SHAP       = _RESOURCE_DIR / "5_aspect_contribution" / "shap.csv"
_DEFAULT_TYPES      = _RESOURCE_DIR / "6_aspect_type" / "aspect_types.csv"
_DEFAULT_OUTPUT_DIR = _RESOURCE_DIR / "7_scenario"
_EPSILON            = 0.001

_SCENARIO_MAP = {
    ("search",     "online"):  "S1",
    ("search",     "offline"): "S2",
    ("experience", "online"):  "S3",
    ("experience", "offline"): "S4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign omnichannel strategy scenarios (S1-S4) to aspects.",
    )
    parser.add_argument("--shap",       type=Path, default=_DEFAULT_SHAP,
                        help="shap_results.csv (default: resource/5_aspect_contribution/shap_results.csv).")
    parser.add_argument("--types",      type=Path, default=_DEFAULT_TYPES,
                        help="aspect_types.csv (default: resource/6_aspect_type/aspect_types.csv).")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR,
                        help="Output directory (default: resource/7_scenario/).")
    parser.add_argument("--epsilon",    type=float, default=_EPSILON,
                        help=f"Min |online_mean_shap - offline_mean_shap| to assign a scenario (default: {_EPSILON}).")
    return parser.parse_args()


def assign_scenario(row: pd.Series, epsilon: float) -> str:
    diff = abs(row["online_mean_shap"] - row["offline_mean_shap"])
    if diff < epsilon:
        return "N/A"
    dom    = "offline" if row["offline_mean_shap"] > row["online_mean_shap"] else "online"
    a_type = str(row["type"]).strip().lower()
    return _SCENARIO_MAP.get((a_type, dom), "Unknown")


def run(args: argparse.Namespace) -> None:
    df_shap  = pd.read_csv(args.shap.resolve())
    df_types = pd.read_csv(args.types.resolve())

    df = df_shap.merge(df_types[["aspect", "type"]], on="aspect", how="left")
    df["scenario"] = df.apply(assign_scenario, axis=1, epsilon=args.epsilon)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scenario_assignment.csv"
    df[["aspect", "scenario"]].to_csv(output_path, index=False, encoding="utf-8-sig")

    print(df["scenario"].value_counts().to_string())
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    run(parse_args())