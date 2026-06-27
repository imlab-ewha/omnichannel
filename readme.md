# Omnichannel

End-to-end pipeline for omnichannel strategy development from Korean product reviews.

## Setup

**1. Install dependencies** (run from `omnichannel/`)

```bash
pip install -r requirements.txt
```

**2. Configure API key**

Create a `.env` file in the `omnichannel/` directory:

```
OPENAI_API_KEY=sk-...
```

**3. Place input file**

```
omnichannel/resource/1_preprocessing/example_review.csv
```

Required columns: `preprocessed_content`, `rating`, `channel` (online / offline)

---

## Quick Start

```bash
cd omnichannel/
python main.py
```

Final output: `resource/8_scenario/scenario_assignment.csv`

---

## Pipeline

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `aspect_extraction.py` | `resource/1_preprocessing/example_review.csv` | `resource/2_aspect_extraction/aspect_extraction.csv` |
| 2 | `aspect_normalization.py` | `resource/2_aspect_extraction/aspect_extraction.csv` | `resource/3_aspect_normalization/normalization_results_*.csv` |
| 3 | `aspect_selection.py` | `resource/3_aspect_normalization/` | `resource/4_aspect_selection/top_aspects.csv`, `top_aspect_reviews.csv` |
| 4 | `sentiment_analysis.py` | `resource/1_preprocessing/example_review.csv` | `resource/5_sentiment_analysis/sentiment_analysis.csv` |
| 5 | `overall_satisfaction_score_combination.py` | `resource/5_sentiment_analysis/sentiment_analysis.csv` | `resource/5_sentiment_analysis/overall_satisfaction.csv` |
| 6 | `random_forest.py` | `resource/4_aspect_selection/`, `resource/5_sentiment_analysis/` | `resource/6_aspect_contribution/models/` |
| 7 | `shap_analysis.py` | `resource/6_aspect_contribution/models/` | `resource/6_aspect_contribution/shap_results.csv` |
| 8 | `aspect_type_determination.py` | `resource/4_aspect_selection/top_aspects.csv` | `resource/7_aspect_type/aspect_types.csv` |
| 9 | `assignment.py` | `resource/6_aspect_contribution/shap_results.csv`, `resource/7_aspect_type/aspect_types.csv` | `resource/8_scenario/scenario_assignment.csv` |

---

## Scripts

### Step 1 — Aspect Extraction

`code/frequent_aspect_mining/aspect_extraction/aspect_extraction.py`

Extracts aspect-opinion-sentiment triplets from Korean reviews using a fine-tuned KcELECTRA model.

| Argument | Default | Description |
|---|---|---|
| `--input` | `resource/1_preprocessing/example_review.csv` | Input review CSV |
| `--output` | `resource/2_aspect_extraction/` | Output directory |
| `--device` | `cpu` | `cuda` or `cpu` |
| `--batch-size` | `32` | Inference batch size |

---

### Step 2 — Aspect Normalization

`code/frequent_aspect_mining/aspect_normalization/aspect_normalization.py`

Normalizes raw aspect expressions into standardized forms via the OpenAI Chat API.

| Argument | Default | Description |
|---|---|---|
| `--input` | `resource/2_aspect_extraction/aspect_extraction.csv` | Input CSV |
| `--output-dir` | `resource/3_aspect_normalization/` | Output directory |
| `--model` | `gpt-4o-mini` | OpenAI model |
| `--chunk-size` | `100` | Pairs per API call |
| `--temperature` | `0.0` | Sampling temperature |
| `--seed` | `42` | Random seed |
| `--sleep-interval` | `3.0` | Seconds between API calls |

---

### Step 3 — Aspect Selection

`code/frequent_aspect_mining/aspect_selection/aspect_selection.py`

Selects the top-k most frequent normalized aspects.

| Argument | Default | Description |
|---|---|---|
| `--input` | `resource/3_aspect_normalization/` | Normalization output dir (picks latest CSV) |
| `--output-dir` | `resource/4_aspect_selection/` | Output directory |
| `--top-k` | `30` | Number of aspects to select |

---

### Step 4 — Sentiment Analysis

`code/aspect_contribution_pairs_mining/overall_satisfaction_score_calculation/sentiment_analysis.py`

Classifies each review as positive / negative using a GRU model. Runs on CPU.

| Argument | Default | Description |
|---|---|---|
| `--input` | `resource/1_preprocessing/example_review.csv` | Input review CSV |
| `--output-dir` | `resource/5_sentiment_analysis/` | Output directory |

Requires `best_model.h5` and `tokenizer.pickle` in the same directory as the script.
Train with `gru_train.py` if not available.

---

### Step 5 — Overall Satisfaction Score

`code/aspect_contribution_pairs_mining/overall_satisfaction_score_calculation/overall_satisfaction_score_combination.py`

Combines star rating and sentiment probability into a continuous satisfaction score (Eq. 1).

| Argument | Default | Description |
|---|---|---|
| `--input` | `resource/5_sentiment_analysis/sentiment_analysis.csv` | Input CSV |
| `--output-dir` | `resource/5_sentiment_analysis/` | Output directory |

---

### Step 6 — Random Forest

`code/aspect_contribution_pairs_mining/aspect_contribution_calculation/random_forest.py`

Trains a Random Forest regressor per channel (online / offline) to predict overall satisfaction from binary aspect-presence features.

| Argument | Default | Description |
|---|---|---|
| `--satisfaction` | `resource/5_sentiment_analysis/overall_satisfaction.csv` | Satisfaction scores |
| `--top-aspects` | `resource/4_aspect_selection/top_aspects.csv` | Top aspect list |
| `--reviews` | `resource/4_aspect_selection/top_aspect_reviews.csv` | Aspect-review pairs |
| `--output-dir` | `resource/6_aspect_contribution/` | Output directory |

---

### Step 7 — SHAP Analysis

`code/aspect_contribution_pairs_mining/aspect_contribution_calculation/shap_analysis.py`

Computes SHAP values from trained RF models and outputs per-aspect mean SHAP per channel.

| Argument | Default | Description |
|---|---|---|
| `--model-dir` | `resource/6_aspect_contribution/models/` | Directory with `{channel}_rf.pkl` files |
| `--output-dir` | `resource/6_aspect_contribution/` | Output directory |

Output columns: `aspect`, `online_mean_abs_shap`, `online_mean_shap`, `offline_mean_abs_shap`, `offline_mean_shap`

---

### Step 8 — Aspect Type Determination

`code/strategy_development/aspect_type_determination/aspect_type_determination.py`

Classifies each aspect as **search** (evaluable before purchase) or **experience** (evaluable only after use) via the OpenAI Chat API.

| Argument | Default | Description |
|---|---|---|
| `--aspects` | `resource/4_aspect_selection/top_aspects.csv` | Aspect list CSV |
| `--output` | `resource/7_aspect_type/aspect_types.csv` | Output CSV |
| `--model` | `gpt-4o-mini` | OpenAI model |
| `--temperature` | `0.0` | Sampling temperature |
| `--seed` | `42` | Random seed |

---

### Step 9 — Scenario Assignment

`code/strategy_development/aspect_scenario_assignment/assignment.py`

Assigns an omnichannel strategy scenario (S1–S4) to each aspect.

| Scenario | Type | Dominant Channel |
|---|---|---|
| S1 | Search | Online |
| S2 | Search | Offline |
| S3 | Experience | Online |
| S4 | Experience | Offline |

| Argument | Default | Description |
|---|---|---|
| `--shap` | `resource/6_aspect_contribution/shap_results.csv` | SHAP results |
| `--types` | `resource/7_aspect_type/aspect_types.csv` | Aspect type results |
| `--output-dir` | `resource/8_scenario/` | Output directory |
| `--epsilon` | `0.001` | Min \|online − offline\| SHAP difference to assign a scenario |

---

## GRU Training (optional)

If `best_model.h5` is not available, train the sentiment model:

```bash
cd code/aspect_contribution_pairs_mining/overall_satisfaction_score_calculation/
python gru_train.py
```

Requires a labeled training CSV. Saves `best_model.h5` and `tokenizer.pickle` to the same directory.
