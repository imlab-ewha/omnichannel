"""
main_fine-tuning.py

Fine-tunes KcELECTRA (AOS extraction) and/or GRU (sentiment analysis).

Usage:
    python main_fine-tuning.py                   # trains both models
    python main_fine-tuning.py --model kc_electra
    python main_fine-tuning.py --model gru
    python main_fine-tuning.py --model all       # same as default

Example data is provided in data/:
    data/kc_electra_train_example.json  (and _dev_)
    data/gru_example.txt

Supply your own data via --train_data / --dev_data (kc_electra)
or --data (gru) to override the defaults.
"""

import argparse
import sys
import time

import src.aspect_extraction as aspect_extraction
import src.sentiment_analysis as sentiment_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune KcELECTRA and/or GRU models."
    )
    parser.add_argument(
        "--model",
        choices=["all", "kc_electra", "gru"],
        required=True,
        help="Which model to fine-tune (default: all)",
    )
    # KcELECTRA overrides
    parser.add_argument("--train_data",          type=str, default=None,
                        help="[kc_electra] Path to training JSON")
    parser.add_argument("--dev_data",            type=str, default=None,
                        help="[kc_electra] Path to dev JSON")
    parser.add_argument("--kc_device",           type=str, default="cuda",
                        help="[kc_electra] Device (cuda / cpu)")
    parser.add_argument("--kc_epochs",           type=int, default=None,
                        help="[kc_electra] Number of training epochs")
    # GRU overrides
    parser.add_argument("--data",                type=str, default=None,
                        help="[gru] Path to tab-separated ratings/reviews file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.model in ("all", "kc_electra"):
        print("\n" + "=" * 60)
        print("  Fine-tuning KcELECTRA (AOS extraction)")
        print("=" * 60)
        kc_args = aspect_extraction.parse_finetune_args([])
        if args.train_data:
            kc_args.train_data = args.train_data
        if args.dev_data:
            kc_args.dev_data = args.dev_data
        kc_args.device = args.kc_device
        if args.kc_epochs:
            kc_args.epochs = args.kc_epochs
        t0 = time.time()
        aspect_extraction.finetune(kc_args)
        print(f"KcELECTRA fine-tuning done in {time.time() - t0:.1f}s")

    if args.model in ("all", "gru"):
        print("\n" + "=" * 60)
        print("  Fine-tuning GRU (sentiment analysis)")
        print("=" * 60)
        gru_args = sentiment_analysis.parse_finetune_gru_args([])
        if args.data:
            gru_args.data = args.data
        t0 = time.time()
        sentiment_analysis.finetune_gru(gru_args)
        print(f"GRU training done in {time.time() - t0:.1f}s")

    print("\nFine-tuning complete.")


if __name__ == "__main__":
    main()
