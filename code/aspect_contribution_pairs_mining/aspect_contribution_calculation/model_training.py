"""
model_training.py

Trains a Random Forest regressor per channel (online / offline) to predict
overall satisfaction from binary aspect-presence (1 / 0) features.

Inputs:
  --satisfaction  resource/4_sentiment_analysis/overall_satisfaction.csv
  --top-aspects   resource/3_aspect_selection/top_aspects.csv
  --reviews       resource/3_aspect_selection/top_aspect_reviews.csv
  --output-dir    resource/5_aspect_contribution/

Outputs:
  resource/5_aspect_contribution/models/{channel}_rf.pkl

Usage:
    python model_training.py [--satisfaction PATH] [--top-aspects PATH]
                             [--reviews PATH] [--output-dir DIR]
"""

import argparse
import pickle
import warnings
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

_SCRIPT_DIR           = Path(__file__).resolve().parent
_RESOURCE_DIR         = _SCRIPT_DIR.parents[2] / "resource"
_DEFAULT_SATISFACTION = _RESOURCE_DIR / "4_sentiment_analysis" / "overall_satisfaction.csv"
_DEFAULT_TOP_ASPECTS  = _RESOURCE_DIR / "3_aspect_selection" / "top_aspects.csv"
_DEFAULT_REVIEWS      = _RESOURCE_DIR / "3_aspect_selection" / "top_aspect_reviews.csv"
_DEFAULT_OUTPUT_DIR   = _RESOURCE_DIR / "5_aspect_contribution"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Random Forest per channel for aspect contribution analysis.",
    )
    parser.add_argument("--satisfaction",    type=Path, default=_DEFAULT_SATISFACTION,
                        help="overall_satisfaction.csv (default: resource/4_sentiment_analysis/overall_satisfaction.csv).")
    parser.add_argument("--top-aspects",     type=Path, default=_DEFAULT_TOP_ASPECTS,
                        help="top_aspects.csv (default: resource/3_aspect_selection/top_aspects.csv).")
    parser.add_argument("--reviews",         type=Path, default=_DEFAULT_REVIEWS,
                        help="top_aspect_reviews.csv (default: resource/3_aspect_selection/top_aspect_reviews.csv).")
    parser.add_argument("--output-dir",      type=Path, default=_DEFAULT_OUTPUT_DIR,
                        help="Output directory (default: resource/5_aspect_contribution/).")
    parser.add_argument("--n-estimators",    type=int,  default=300,
                        help="Number of trees in the forest (default: 300).")
    parser.add_argument("--max-depth",       type=int,  default=8,
                        help="Maximum depth of each tree (default: 8).")
    parser.add_argument("--min-samples-leaf",type=int,  default=5,
                        help="Minimum samples required at a leaf node (default: 5).")
    parser.add_argument("--random-state",    type=int,  default=42,
                        help="Random seed for reproducibility (default: 42).")
    return parser.parse_args()


def build_feature_matrix(
    df_sat: pd.DataFrame,
    df_reviews: pd.DataFrame,
    top_aspects: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    y = df_sat.groupby("preprocessed_content")["overall_satisfaction"].mean()

    df_rev = df_reviews[df_reviews["preprocessed_content"].isin(y.index)].copy()
    df_rev = df_rev[df_rev["normalized_aspect"].isin(top_aspects)]
    df_rev = df_rev[["preprocessed_content", "normalized_aspect"]].drop_duplicates()
    df_rev["value"] = 1

    one_hot = df_rev.pivot_table(
        index="preprocessed_content", columns="normalized_aspect",
        values="value", aggfunc="first", fill_value=0,
    )
    for asp in top_aspects:
        if asp not in one_hot.columns:
            one_hot[asp] = 0
    one_hot = one_hot[top_aspects]

    final = one_hot.join(y, how="inner").dropna()
    return final[top_aspects], final["overall_satisfaction"]


def run(args: argparse.Namespace) -> None:
    df_sat      = pd.read_csv(args.satisfaction.resolve())
    top_aspects = pd.read_csv(args.top_aspects.resolve())["normalized_aspect"].tolist()
    df_reviews  = pd.read_csv(args.reviews.resolve())

    model_dir = args.output_dir.resolve() / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for channel in ["online", "offline"]:
        print(f"\n[{channel}]")
        df_ch = df_sat[df_sat["channel"] == channel]

        X, y = build_feature_matrix(df_ch, df_reviews, top_aspects)
        print(f"  Samples: {len(X)}")

        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
        )
        model.fit(X, y)

        model_path = model_dir / f"{channel}_rf.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "X": X, "X_columns": list(X.columns)}, f)
        print(f"  Model saved: {model_path}")

    print("\nDone.")


if __name__ == "__main__":
    run(parse_args())
