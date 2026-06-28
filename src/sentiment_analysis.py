"""
sentiment_analysis.py

Runs binary (positive / negative) sentiment analysis on preprocessed Korean
product reviews using a fine-tuned Keras GRU model and MeCab tokenizer.

Usage:
    python -m src.sentiment_analysis [--input PATH] [--output-dir DIR]
"""

import argparse
import logging
import os
import pickle
import re
import time
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
logging.getLogger("absl").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from mecab import MeCab
from sklearn.metrics import classification_report, f1_score  # training only
from sklearn.model_selection import train_test_split          # training only
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint  # training only
from tensorflow.keras.layers import Dense, Embedding, GRU    # training only
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer    # training only

_ROOT               = Path(__file__).resolve().parents[1]
_MODEL_PATH         = _ROOT / "checkpoints" / "gru" / "sentiment_analysis_model.h5"
_TOKENIZER_PATH     = _ROOT / "checkpoints" / "gru" / "sentiment_analysis_tokenizer.pkl"
_DEFAULT_INPUT      = _ROOT / "data" / "example_review.csv"
_DEFAULT_OUTPUT_DIR = _ROOT / "outputs" / "sentiment_analysis"

_MAX_LEN = 80


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sentiment analysis on preprocessed Korean product reviews.",
    )
    parser.add_argument("--input",      type=Path, default=_DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    return parser.parse_args(args)


def _predict(text: str, model, tokenizer, mecab: MeCab) -> tuple[str, float]:
    text    = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", str(text))
    tokens  = [token.surface for token in mecab.parse(text)]
    encoded = tokenizer.texts_to_sequences([tokens])
    padded  = pad_sequences(encoded, maxlen=_MAX_LEN)
    score   = round(float(model.predict(padded, verbose=0)), 4)
    return ("positive" if score > 0.5 else "negative"), score


def run(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GRU inference runs on CPU
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
    df.to_csv(output_dir / "sentiment_analysis.csv", index=False, encoding="utf-8-sig")

    print(f"Sentiment analysis complete. {len(df)} reviews processed.")
    print(f"Saved: {output_dir / 'sentiment_analysis.csv'}")


# ---------------------------------------------------------------------------
# GRU training
# ---------------------------------------------------------------------------

class _EpochTimer(Callback):
    """Keras callback that prints elapsed time per epoch and total."""

    def __init__(self):
        self.total_start = None

    def on_train_begin(self, logs=None):
        self.total_start = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        print(f"  Epoch {epoch + 1} time: {time.time() - self._start:.1f}s")

    def on_train_end(self, logs=None):
        total = time.time() - self.total_start
        print(f"Total training time: {int(total // 60)}m {int(total % 60)}s")


_TRAIN_EMBED_DIM   = 100
_TRAIN_HIDDEN      = 128
_TRAIN_EPOCHS      = 15
_TRAIN_BATCH_SIZE  = 64
_TRAIN_MIN_FREQ    = 2
_TRAIN_TEST_SIZE   = 0.25
_TRAIN_SEED        = 42
_TRAIN_RESULTS_DIR = _ROOT / "checkpoints" / "gru"
_TRAIN_DEFAULT_DATA = _ROOT / "data" / "gru_example.txt"

_STOPWORDS = {
    '도','는','다','의','가','이','은','한','에','하','고',
    '을','를','인','듯','과','와','네','들','지','임','게',
}


def parse_finetune_gru_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRU sentiment classifier.")
    parser.add_argument("--data", type=str, default=str(_TRAIN_DEFAULT_DATA),
                        help="Tab-separated file with columns: ratings, reviews (no header)")
    return parser.parse_args(args)


def finetune_gru(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {data_path}\n"
            "Provide a tab-separated file with columns: ratings, reviews (no header)."
        )

    mecab = MeCab()
    timestamp = time.strftime("%y%m%d_%H%M%S")

    df = pd.read_table(str(data_path), names=["ratings", "reviews"])
    print(f"Loaded {len(df)} reviews.")
    df["label"] = np.select([df.ratings > 3], [1], default=0)
    df.drop_duplicates(subset=["reviews"], inplace=True)

    train_df, test_df = train_test_split(df, test_size=_TRAIN_TEST_SIZE, random_state=_TRAIN_SEED)

    def _preprocess(frame):
        frame = frame.copy()
        frame["reviews"] = frame["reviews"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)
        frame["reviews"].replace("", np.nan, inplace=True)
        frame.dropna(inplace=True)
        frame["tokenized"] = frame["reviews"].apply(
            lambda x: [t.surface for t in mecab.parse(x) if t.surface not in _STOPWORDS]
        )
        return frame

    train_df = _preprocess(train_df)
    test_df  = _preprocess(test_df)

    tok = Tokenizer()
    tok.fit_on_texts(train_df["tokenized"])
    rare  = sum(1 for v in tok.word_counts.values() if v < _TRAIN_MIN_FREQ)
    vocab = len(tok.word_index) - rare + 2
    tok   = Tokenizer(vocab, oov_token="OOV")
    tok.fit_on_texts(train_df["tokenized"])
    print(f"Vocabulary size: {vocab}")

    _TRAIN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tok_path = _TRAIN_RESULTS_DIR / f"sentiment_analysis_tokenizer_{timestamp}.pkl"
    with open(tok_path, "wb") as f:
        pickle.dump(tok, f)
    print(f"Tokenizer saved: {tok_path}")

    X_train = pad_sequences(tok.texts_to_sequences(train_df["tokenized"]), maxlen=_MAX_LEN)
    X_test  = pad_sequences(tok.texts_to_sequences(test_df["tokenized"]),  maxlen=_MAX_LEN)
    Y_train = train_df["label"].values
    Y_test  = test_df["label"].values

    model = Sequential([
        Embedding(vocab, _TRAIN_EMBED_DIM),
        GRU(_TRAIN_HIDDEN),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])

    model_path = str(_TRAIN_RESULTS_DIR / f"sentiment_analysis_model_{timestamp}.h5")
    model.fit(
        X_train, Y_train,
        epochs=_TRAIN_EPOCHS, batch_size=_TRAIN_BATCH_SIZE,
        validation_split=0.2,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=4, verbose=1),
            ModelCheckpoint(model_path, monitor="val_acc", mode="max", save_best_only=True, verbose=1),
            _EpochTimer(),
        ],
    )

    loaded = load_model(model_path)
    _loss, acc = loaded.evaluate(X_test, Y_test, verbose=0)
    Y_pred = (loaded.predict(X_test, verbose=0) > 0.5).astype(int)
    print(f"\nTest accuracy: {acc:.4f}")
    print(f"Test F1: {f1_score(Y_test, Y_pred, zero_division=0):.4f}")
    print(classification_report(
        Y_test, Y_pred, labels=[0, 1],
        target_names=["negative", "positive"], zero_division=0,
    ))
    print(f"Model saved: {model_path}")


if __name__ == "__main__":
    run(parse_args())
