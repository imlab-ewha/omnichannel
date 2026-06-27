"""
aspect_normalization.py

Normalizes raw aspect expressions from customer reviews into standardized
forms using the OpenAI Chat API with in-context learning.

Inputs:
  --input       resource/1_aspect_extraction/aspect_extraction.csv

Outputs:
  resource/2_aspect_normalization/normalization_{timestamp}.csv
  resource/2_aspect_normalization/normalization_{timestamp}.json

Usage:
    python aspect_normalization.py [--input PATH] [--output-dir DIR]
                                   [--model MODEL] [--chunk-size N]
                                   [--temperature T] [--seed N]
                                   [--sleep-interval S]
"""

import argparse
import json
import os
import time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_SCRIPT_DIR         = Path(__file__).resolve().parent
_RESOURCE_DIR       = _SCRIPT_DIR.parents[2] / "resource"
_DEFAULT_INPUT      = _RESOURCE_DIR / "1_aspect_extraction" / "aspect_extraction.csv"
_DEFAULT_OUTPUT_DIR = _RESOURCE_DIR / "2_aspect_normalization"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
Role description: You are an expert in aspect normalization for product review analysis.
Your task is to standardize variant aspect expressions into unified forms.

Task description: Normalize the provided aspects according to the guidelines below.

Normalization guidelines:
Given a list of (aspect, sentence) pairs, normalize each aspect as follows:

Step 1 — Group aspects
Group aspects if they meet either of the following:
- Synonyms: Different expressions with the same meaning in the context of product reviews.
    - Example: ["촉촉함", "보습력", "보습감"]
- Hierarchical: A specific aspect that belongs to a broader concept.
    - Example: ["바닐라 향", "우디 향", "시트러스 향"]
    - A single aspect cannot form a hierarchical group on its own. Do not generalize an aspect in isolation.

Step 2 — Select representative for each group
- Synonym groups: Choose the most general and commonly used expression from within the group.
    - Suffix removal is allowed: "쿨링감", "쿨링 효과" → "쿨링", "지성 피부" → "지성"
- Hierarchical groups: Replace with the broader term. The broader term does not need to appear in the input.

Step 3 — Handle ungrouped aspects
If an aspect does not belong to any group, **return "n/a"**.

Avoid Over-Generalization
- Do not use abstract meta-categories like "상태", "제형", "효과", or "제품" as normalized forms.

Task input: A list of (aspect, sentence) pairs:
[("aspect1", "sentence1"), ("aspect2", "sentence2"), ...]

Task output: Return only a JSON object mapping each input aspect to its normalized form.
Do not include explanations or additional text.

Demonstrations:
Input:
[("바닐라 향", "바닐라 향이 너무 달콤하고 사랑스러워요."),
("레몬 향", "레몬 향이 오래 지속되어서 정말 마음에 듭니다."),
("보습력", "보습력이 아주 만족스럽습니다."),
("촉촉함", "촉촉함이 너무 좋아서 친구에게도 선물했어요."),
("패드 크기", "패드 크기가 큼직해서 아주 마음에 듭니다."),
("사이즈", "생각했던 것보다 사이즈가 작아서 조금 아쉬웠어요."),
("지성 피부", "제가 지성 피부라서 유분기가 금방 올라와요."),
("지성인", "제가 지성인이어서 이 패드가 잘 맞네요."),
("아연", "아연 성분이 함유되어 있어서 좋습니다.")]

Output:
{"바닐라 향": "향", "레몬 향": "향", "보습력": "보습", "촉촉함": "보습", "패드 크기": "사이즈", "사이즈": "사이즈", "지성 피부": "지성", "지성인": "지성", "아연": "n/a"}
"""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize aspects via the OpenAI chat API.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help="Input CSV file containing 'aspect' and 'sentence' columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory where output CSV and JSON files will be written.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default='gpt-4o-mini',
        help="OpenAI chat model identifier.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Number of (aspect, sentence) pairs sent per API call.",
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
    parser.add_argument(
        "--sleep-interval",
        type=float,
        default=3.0,
        help="Seconds to sleep between API calls.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def normalize_chunk(
    client: OpenAI,
    pairs: list[tuple[str, str]],
    *,
    model: str,
    temperature: float,
    seed: int,
) -> dict[str, str]:
    """Send one batch of (aspect, sentence) pairs to the API and return the normalized mapping."""
    response = client.chat.completions.create(
        model=model,
        seed=seed,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Normalize these (aspect, sentence) pairs: {pairs}"},
        ],
    )
    content = response.choices[0].message.content.strip()
    return json.loads(content)


def run(args: argparse.Namespace) -> None:
    timestamp = time.strftime("%y%m%d_%H%M%S")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path  = output_dir / f"normalization_{timestamp}.csv"
    output_json_path = output_dir / f"normalization_{timestamp}.json"

    df = pd.read_csv(args.input.resolve())
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    all_normalized: dict[str, str] = {}
    total_chunks = -(-len(df) // args.chunk_size)

    for i in range(0, len(df), args.chunk_size):
        chunk_df  = df.iloc[i : i + args.chunk_size]
        chunk_num = (i // args.chunk_size) + 1
        pairs     = list(chunk_df[["aspect", "sentence"]].itertuples(index=False, name=None))

        try:
            parsed = normalize_chunk(
                client, pairs,
                model=args.model,
                temperature=args.temperature,
                seed=args.seed,
            )
            all_normalized.update(parsed)
        except Exception as e:
            print(f"[Chunk {chunk_num}/{total_chunks}] Error: {e}")

        time.sleep(args.sleep_interval)

    df["normalized_aspect"] = df["aspect"].map(all_normalized)
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_normalized, f, ensure_ascii=False, indent=4)

    print(f"Normalization complete. {len(all_normalized)} aspects normalized.")
    print("\nSample results:")
    for aspect, normalized in list(all_normalized.items())[:5]:
        print(f"  {aspect} -> {normalized}")
    print(f"\nSaved:")
    print(f"  CSV  : {output_csv_path}")
    print(f"  JSON : {output_json_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run(parse_args())
