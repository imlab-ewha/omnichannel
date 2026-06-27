"""
aspect_contribution_calculation.py

Computes SHAP values from trained Random Forest models (one per channel)
and outputs per-aspect mean absolute SHAP and mean SHAP for online and offline.

Input:
  resource/5_aspect_contribution/models/{online,offline}_rf.pkl

Output:
  resource/5_aspect_contribution/shap.csv
  Columns: aspect, online_mean_abs_shap, online_mean_shap,
                   offline_mean_abs_shap, offline_mean_shap

Usage:
    python aspect_contribution_calculation.py [--model-dir DIR] [--output-dir DIR]
"""

import argparse
import pickle
import warnings
from pathlib import Path

import pandas as pd
import shap

warnings.filterwarnings("ignore")

_SCRIPT_DIR         = Path(__file__).resolve().parent
_RESOURCE_DIR       = _SCRIPT_DIR.parents[2] / "resource"
_DEFAULT_MODEL_DIR  = _RESOURCE_DIR / "5_aspect_contribution" / "models"
_DEFAULT_OUTPUT_DIR = _RESOURCE_DIR / "5_aspect_contribution"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-channel SHAP values from trained Random Forest models.",
    )
    parser.add_argument("--model-dir",  type=Path, default=_DEFAULT_MODEL_DIR,
                        help="Directory containing {channel}_rf.pkl files.")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR,
                        help="Directory where shap.csv will be written.")
    return parser.parse_args()


def compute_channel_shap(model_path: Path) -> pd.DataFrame:
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    X     = saved["X"]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return pd.DataFrame(shap_values, columns=X.columns)


def run(args: argparse.Namespace) -> None:
    model_dir  = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    channel_dfs = {}

    for channel in ["online", "offline"]:
        model_path = model_dir / f"{channel}_rf.pkl"
        if not model_path.exists():
            print(f"Model not found: {model_path} — skipping.")
            continue

        print(f"Computing SHAP for {channel}...")
        shap_df = compute_channel_shap(model_path)

        channel_dfs[channel] = pd.DataFrame({
            f"{channel}_mean_abs_shap": shap_df.abs().mean().round(4),
            f"{channel}_mean_shap":     shap_df.mean().round(4),
        })

    if not channel_dfs:
        print("No models found. Run model_training.py first.")
        return

    result = pd.concat(channel_dfs.values(), axis=1)
    result.index.name = "aspect"
    result = result.reset_index()

    output_path = output_dir / "shap.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    run(parse_args())
