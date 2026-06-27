"""
sentiment_analysis.py

Runs binary (positive / negative) sentiment analysis on preprocessed Korean
product reviews using a fine-tuned Keras model and MeCab tokenizer.

Usage:
    python sentiment_analysis.py [--input PATH] [--output-dir DIR]
"""

import argparse
import logging
import os
import pickle
import re
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
logging.getLogger("absl").setLevel(logging.ERROR)

import pandas as pd
from mecab import MeCab
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

_SCRIPT_DIR         = Path(__file__).resolve().parent
_RESOURCE_DIR       = _SCRIPT_DIR.parents[2] / "resource"
_DEFAULT_INPUT      = _RESOURCE_DIR / "example_review.csv"
_DEFAULT_OUTPUT_DIR = _RESOURCE_DIR / "4_sentiment_analysis"
_MODEL_PATH         = _SCRIPT_DIR / "best_model.h5"
_TOKENIZER_PATH     = _SCRIPT_DIR / "tokenizer.pickle"

_MAX_LEN = 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sentiment analysis on preprocessed Korean product reviews.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help="Input CSV containing a 'preprocessed_content' column.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory where the output CSV will be written.",
    )
    return parser.parse_args()


def _predict(text: str, model, tokenizer, mecab: MeCab) -> tuple[str, float]:
    text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", str(text))
    tokens = [token.surface for token in mecab.parse(text)]
    encoded = tokenizer.texts_to_sequences([tokens])
    padded = pad_sequences(encoded, maxlen=_MAX_LEN)
    score = round(float(model.predict(padded, verbose=0)), 4)
    sentiment = "positive" if score > 0.5 else "negative"
    return sentiment, score


def run(args: argparse.Namespace) -> None:
    print("Loading model...")
    model = load_model(str(_MODEL_PATH))
    with open(_TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    mecab = MeCab()

    df = pd.read_csv(args.input.resolve())

    sentiments, pos_probs = [], []
    for idx, text in enumerate(df["preprocessed_content"]):
        try:
            sentiment, pos_prob = _predict(text, model, tokenizer, mecab)
        except Exception as e:
            print(f"  Error at row {idx}: {e}")
            sentiment, pos_prob = "", 0.0
        sentiments.append(sentiment)
        pos_probs.append(pos_prob)

    df["sentiment"]              = sentiments
    df["positivity_probability"] = pos_probs

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sentiment_analysis.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Sentiment analysis complete. {len(df)} reviews processed.")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    run(parse_args())
