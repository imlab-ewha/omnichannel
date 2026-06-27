# Omnichannel

## Setup

**1. Install dependencies**

```bash
pip install -r omnichannel/requirements.txt
```

**2. Configure API key**

Create a `.env` file in the `omnichannel/` directory:

```
OPENAI_API_KEY=sk-...
```

Both scripts automatically load this file via `python-dotenv`.

**3. Run from the `omnichannel/code/` directory**

```bash
cd omnichannel/code/

python frequent_aspect_mining/aspect_normalization/aspect_normalization.py --input ...
python strategy_development/aspect_type_determination/aspect_type_determination.py --aspects ...
```

---

## Scripts

### 1. `frequent_aspect_mining/aspect_normalization.py`

Normalizes raw aspect expressions from customer reviews into standardized forms using the OpenAI Chat API.

**Input**

A CSV file with the following columns:

| Column | Description |
|---|---|
| `aspect` | Raw aspect expression extracted from a review |
| `sentence` | The review sentence the aspect appeared in |

**Output**

Two files written to `--output-dir`, timestamped at runtime:

| File | Description |
|---|---|
| `normalization_YYMMDD_HHMMSS.csv` | Input CSV with an added `normalized_aspect` column |
| `normalization_YYMMDD_HHMMSS.json` | Flat mapping of `{raw_aspect: normalized_aspect}` |

**Arguments**

| Argument | Default | Description |
|---|---|---|
| `--input` | `resource/1_aspect_extraction/aspect_extraction.csv` | Path to input CSV |
| `--output-dir` | `resource/2_aspect_normalization/` | Directory for output files |
| `--model` | `gpt-4o-mini` | OpenAI model |
| `--chunk-size` | `100` | Number of pairs per API call |
| `--temperature` | `0.0` | Sampling temperature |
| `--seed` | `42` | Random seed for reproducibility |
| `--sleep-interval` | `3.0` | Seconds to wait between API calls |

**Example**

```bash
cd omnichannel/code/
python frequent_aspect_mining/aspect_normalization/aspect_normalization.py \
  --input data/aspects.csv \
  --output-dir results/normalization \
  --model gpt-4o-mini \
  --temperature 0 \
  --seed 42 \
```

---

### 2. `strategy_development/aspect_type_determination.py`

Determines whether each aspect is a **search** or an **experience**, based on whether it can be evaluated before or only after using the product.

**Input**

Accepts either format:

| Format | Requirement |
|---|---|
| `.csv` | Must contain an `aspect` column |

**Output**

A single CSV at the path specified by `--output`:

| Column | Description |
|---|---|
| `aspect` | Aspect name |
| `type` | `search` or `experience` |
| `reason` | One-sentence explanation in Korean |

**Arguments**

| Argument | Required | Description |
|---|---|---|
| `--aspects` | Yes | Path to aspects file (`.csv`) |
| `--output` | Yes | Output CSV file path |
| `--model` | Yes | OpenAI model (e.g., `gpt-4o-mini`) |
| `--temperature` | No | Sampling temperature |
| `--seed` | No | Random seed for reproducibility |

**Example**

```bash
cd omnichannel/code/
python strategy_development/aspect_type_determination/aspect_type_determination.py \
  --aspects results/normalization/normalization_260101_120000.csv \
  --output results/type_determination/aspect_types.csv \
  --model gpt-4o-mini \
  --temperature 0 \
  --seed 42
```
 
