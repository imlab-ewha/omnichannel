"""
aspect_extraction.py

Extracts Aspect-Opinion-Sentiment (AOS) triplets from Korean product reviews
using a fine-tuned KcELECTRA-based model.

Usage:
    python -m src.aspect_extraction [--input PATH] [--output PATH] [--batch-size N]
"""

import argparse
import logging
import math
import os
import re
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Text

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import dask
import torch
import torch.nn
from kiwipiepy import Kiwi
import pandas as pd
from transformers import AutoConfig, AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(module)s:%(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_ROOT           = Path(__file__).resolve().parents[1]
_CHECKPOINT_DIR = _ROOT / "checkpoints" / "kc_electra"
_DEFAULT_INPUT  = _ROOT / "data" / "example_review.csv"
_DEFAULT_OUTPUT = _ROOT / "outputs" / "aspect_extraction"

_MAX_SEQ_LEN   = 150
_BERT_FEAT_DIM = 768
_DROPOUT       = 0.3
_N_HOPS        = 1
_CLASS_NUM     = 5

_TAIL_TAGS = {
    "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC",
    "EF", "EC", "SF", "SP", "SS", "SE", "SW",
}
_COPULA_RE = re.compile(r"(이에요|이었어요|이었습니다|입니다|이야|이었|이고|이라|이다)[.!?]*$")


# ---------------------------------------------------------------------------
# AOS decoding helpers
# ---------------------------------------------------------------------------

def get_spans(tags, length, token_range, span_type, ignore_index=-1):
    spans = []
    start = -1
    for i in range(length):
        l, _ = token_range[i]
        if tags[l][l] == ignore_index:
            continue
        elif tags[l][l] == span_type:
            if start == -1:
                start = i
        else:
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


def find_triplet(tags, aspect_spans, opinion_spans, token_ranges, probs):
    triplets = []
    for al, ar in aspect_spans:
        for pl, pr in opinion_spans:
            tag_num  = [0] * 6
            prob_lst = []
            for i in range(al, ar + 1):
                for j in range(pl, pr + 1):
                    a_start = token_ranges[i][0]
                    o_start = token_ranges[j][0]
                    if al < pl:
                        tag_num[int(tags[a_start][o_start])] += 1
                        prob_lst.append(float(probs[a_start][o_start]))
                    else:
                        tag_num[int(tags[o_start][a_start])] += 1
                        prob_lst.append(float(probs[o_start][a_start]))

            if sum(tag_num[3:]) == 0:
                continue

            sentiment_prob = sum(prob_lst) / len(prob_lst)

            if tag_num[4] > tag_num[3] and tag_num[4] > tag_num[5]:
                sentiment = "positive"
            elif tag_num[3] > tag_num[4] and tag_num[3] > tag_num[5]:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            triplets.append([al, ar, pl, pr, sentiment, sentiment_prob])

    return triplets


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class MultiInferBert(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bert = AutoModel.from_config(AutoConfig.from_pretrained(args.bert_model_path))
        self.cls_linear     = torch.nn.Linear(args.bert_feature_dim * 2, args.class_num)
        self.feature_linear = torch.nn.Linear(
            args.bert_feature_dim * 2 + args.class_num * 3, args.bert_feature_dim * 2
        )
        self.dropout_output = torch.nn.Dropout(p=args.dropout)

    def multi_hops(self, features, mask, k):
        max_length = features.shape[1]
        mask = mask[:, :max_length]
        mask_a = mask.unsqueeze(1).expand([-1, max_length, -1])
        mask_b = mask.unsqueeze(2).expand([-1, -1, max_length])
        mask = mask_a * mask_b
        mask = torch.triu(mask).unsqueeze(3).expand([-1, -1, -1, self.args.class_num])

        logits_list = []
        logits = self.cls_linear(features)
        logits_list.append(logits)

        for _ in range(k):
            probs  = logits
            logits = probs * mask
            logits_a = torch.max(logits, dim=1)[0]
            logits_b = torch.max(logits, dim=2)[0]
            logits = torch.cat([logits_a.unsqueeze(3), logits_b.unsqueeze(3)], dim=3)
            logits = torch.max(logits, dim=3)[0]
            logits = logits.unsqueeze(2).expand([-1, -1, max_length, -1])
            logits_T = logits.transpose(1, 2)
            logits = torch.cat([logits, logits_T], dim=3)

            new_features = torch.cat([features, logits, probs], dim=3)
            features = self.feature_linear(new_features)
            logits   = self.cls_linear(features)
            logits_list.append(logits)

        return logits_list[-1]

    def forward(self, tokens, masks):
        bert_feature = self.bert(tokens, masks)["last_hidden_state"]
        bert_feature = self.dropout_output(bert_feature)
        bert_feature = bert_feature.unsqueeze(2).expand([-1, -1, self.args.max_sequence_len, -1])
        bert_feature_T = bert_feature.transpose(1, 2)
        features = torch.cat([bert_feature, bert_feature_T], dim=3)
        del bert_feature, bert_feature_T
        torch.cuda.empty_cache()

        logits = self.multi_hops(features, masks, self.args.nhops)
        del features, masks
        torch.cuda.empty_cache()
        return logits


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

@dataclass
class TaggingInstance:
    id:                  int
    sentence:            Text
    sen_length:          int
    token_range:         List[Any]
    bert_tokens:         List[Any]
    bert_tokens_padding: torch.Tensor
    mask:                torch.Tensor
    origin_review:       Text


def _model_config(device: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        model_dir        = str(_CHECKPOINT_DIR / "aspect_extraction_model.pt"),
        bert_model_path  = str(_CHECKPOINT_DIR),
        max_sequence_len = _MAX_SEQ_LEN,
        bert_feature_dim = _BERT_FEAT_DIM,
        class_num        = _CLASS_NUM,
        dropout          = _DROPOUT,
        nhops            = _N_HOPS,
        device           = device,
    )


class ReviewProcessor:
    def __init__(self, *, device: str = "cpu", batch_size: int = 50) -> None:
        if device == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        config = _model_config(device)
        self.device     = device
        self.batch_size = batch_size
        self.model      = MultiInferBert(config).to(device)
        self.model.load_state_dict(
            torch.load(config.model_dir, map_location=device), strict=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")
        self.kiwi      = Kiwi()
        logger.info("ReviewProcessor ready.")

    def _flatten(self, lst):
        return [item for sublist in lst for item in sublist]

    def preprocessing(self, idx: int, record: Dict[Text, Any]) -> List[Any]:
        review = record.get("corrected_text")
        if not review:
            return []
        sentences = list(dict.fromkeys(s.text for s in self.kiwi.split_into_sents(review)))
        return [[idx, s, review] for s in sentences]

    def get_tagging_instance(self, review: List[Any]):
        review_id, sentence, origin_review = review[0], review[1], review[2]
        tokens  = sentence.strip().split()
        sen_len = len(tokens)

        bert_tokens         = self.tokenizer.encode(sentence, max_length=_MAX_SEQ_LEN)
        bert_tokens_padding = torch.zeros(_MAX_SEQ_LEN).long()
        mask                = torch.zeros(_MAX_SEQ_LEN)

        for i, tok in enumerate(bert_tokens):
            bert_tokens_padding[i] = tok
        mask[: len(bert_tokens)] = 1

        token_range  = []
        token_start  = 1
        for w in tokens:
            token_end = token_start + len(self.tokenizer.encode(w, add_special_tokens=False))
            token_range.append([token_start, token_end - 1])
            token_start = token_end

        if len(bert_tokens) != token_range[-1][-1] + 2:
            return None

        return TaggingInstance(
            review_id, sentence, sen_len,
            token_range, bert_tokens, bert_tokens_padding, mask, origin_review,
        )

    def _get_batch(self, index: int, instances: List[TaggingInstance]):
        start = index * self.batch_size
        end   = min(start + self.batch_size, len(instances))
        batch = instances[start:end]

        sentence_ids    = [inst.id                  for inst in batch]
        sentences       = [inst.sentence            for inst in batch]
        sen_lengths     = [inst.sen_length          for inst in batch]
        token_ranges    = [inst.token_range         for inst in batch]
        bert_tokens_pad = [inst.bert_tokens_padding for inst in batch]
        bert_tokens_raw = [inst.bert_tokens         for inst in batch]
        masks           = [inst.mask                for inst in batch]

        tokens_tensor = torch.stack(bert_tokens_pad).to(self.device)
        masks_tensor  = torch.stack(masks).to(self.device)

        return sentence_ids, sentences, tokens_tensor, masks_tensor, sen_lengths, token_ranges, bert_tokens_raw

    def prediction(self, instances: List[TaggingInstance]) -> Dict[Text, List[Any]]:
        valid_instances = [inst for inst in instances if inst is not None]
        batch_count     = math.ceil(len(valid_instances) / self.batch_size)
        id_to_inst      = {inst.id: inst for inst in valid_instances}

        all_ids = []; all_preds = []; all_sen_lengths = []
        all_token_ranges = []; all_tokens_raw = []; all_probs = []
        all_sentences = []; all_origin_reviews = []

        self.model.eval()
        with torch.no_grad():
            for i in range(batch_count):
                try:
                    (
                        sentence_ids, sentences, bert_tokens, masks,
                        sen_lengths, token_ranges, bert_tokens_raw,
                    ) = self._get_batch(i, valid_instances)
                except Exception as e:
                    logger.warning(f"Batch {i} skipped: {e}")
                    continue

                preds = self.model(bert_tokens, masks)
                probs = torch.sigmoid(preds)
                preds = torch.argmax(preds, dim=3)
                probs = torch.max(probs, dim=3)[0]

                all_ids.extend(sentence_ids)
                all_preds.append(preds.cpu())
                all_sen_lengths.extend(sen_lengths)
                all_token_ranges.extend(token_ranges)
                all_tokens_raw.extend(bert_tokens_raw)
                all_probs.append(probs.cpu())
                all_sentences.extend(sentences)
                all_origin_reviews.extend([
                    id_to_inst[sid].origin_review if sid in id_to_inst else ""
                    for sid in sentence_ids
                ])

                del sentence_ids, sentences, bert_tokens, masks, sen_lengths
                del token_ranges, bert_tokens_raw, preds, probs
                torch.cuda.empty_cache()

        return {
            "all_ids":            all_ids,
            "all_preds":          torch.cat(all_preds, dim=0).cpu().tolist(),
            "all_sen_lengths":    all_sen_lengths,
            "all_token_ranges":   all_token_ranges,
            "all_tokens_raw":     all_tokens_raw,
            "all_probs":          torch.cat(all_probs, dim=0).cpu().tolist(),
            "all_sentences":      all_sentences,
            "all_origin_reviews": all_origin_reviews,
        }

    def get_aos(self, predictions: Dict[Text, List[Any]]) -> List[Any]:
        aos_set = []
        for sid, preds, sen_length, token_ranges, tokens_raw, probs, sentence, origin_review in zip(
            predictions["all_ids"],
            predictions["all_preds"],
            predictions["all_sen_lengths"],
            predictions["all_token_ranges"],
            predictions["all_tokens_raw"],
            predictions["all_probs"],
            predictions["all_sentences"],
            predictions["all_origin_reviews"],
        ):
            predicted_tuples = find_triplet(
                preds,
                get_spans(preds, sen_length, token_ranges, 1),
                get_spans(preds, sen_length, token_ranges, 2),
                token_ranges,
                probs,
            )
            for p_tuple in predicted_tuples:
                a_start    = token_ranges[p_tuple[0]][0]
                a_end      = token_ranges[p_tuple[1]][-1]
                raw_aspect = self.tokenizer.decode(tokens_raw[a_start: a_end + 1])
                aspect_pos = [[t.form, str(t.tag)] for t in self.kiwi.tokenize(raw_aspect)]
                aos_set.append({
                    "sid":                  sid,
                    "preprocessed_content": origin_review,
                    "sentence":             sentence,
                    "raw_aspect":           raw_aspect,
                    "aspect_pos":           aspect_pos,
                })
        return aos_set

    def _extract_aspect(self, records: List[Dict[Text, Any]]) -> List[Dict[Text, Any]]:
        for record in records:
            pos  = record.get("aspect_pos", [])
            raw  = record.get("raw_aspect", "")
            core = _COPULA_RE.sub("", raw).strip()
            for form, tag in reversed(pos):
                if tag in _TAIL_TAGS or tag == "VCP":
                    if core.endswith(form):
                        core = core[: -len(form)].strip()
                else:
                    break
            record["aspect"]     = core
            record["aspect_pos"] = str(pos)
        return records

    def tagging(self, records: List[Dict[Text, Any]]) -> List[Dict[Text, Any]]:
        logger.info("Prepare tagging.")
        preprocessor = dask.delayed(self.preprocessing) if len(records) > 50 else self.preprocessing
        preprocessed = [preprocessor(i, r) for i, r in enumerate(records)]
        preprocessed = dask.compute(*preprocessed)
        preprocessed = self._flatten(preprocessed)
        logger.info(f"Preprocessing done: {len(preprocessed)} sentences.")

        instances = [self.get_tagging_instance(x) for x in preprocessed]
        logger.info(f"Instance creation done: {len(instances)} instances.")

        result_dict = self.prediction(instances)
        result_aos  = self.get_aos(result_dict)
        result_aos  = self._extract_aspect(result_aos)
        logger.info(f"Tagging done: {len(result_aos)} triplets.")
        return result_aos


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract AOS triplets from Korean product reviews.",
    )
    parser.add_argument("--input",      type=Path, default=_DEFAULT_INPUT,
                        help="Input CSV with 'preprocessed_content' column.")
    parser.add_argument("--output",     type=Path, default=_DEFAULT_OUTPUT,
                        help="Output directory.")
    parser.add_argument("--batch-size", type=int,  default=50,
                        help="Number of sentences per model forward pass.")
    return parser.parse_args(args)


def run(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input.resolve())
    df = df.rename(columns={"preprocessed_content": "corrected_text"})
    records = df.to_dict(orient="records")

    processor = ReviewProcessor(device="cuda", batch_size=args.batch_size)
    results   = processor.tagging(records)

    output_path = args.output.resolve()
    if output_path.suffix != ".csv":
        output_path = output_path / "aspect_extraction.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame(results)[
        ["sid", "preprocessed_content", "sentence", "raw_aspect", "aspect_pos", "aspect"]
    ]
    result_df = result_df.drop_duplicates(subset=["sid", "raw_aspect"]).reset_index(drop=True)
    result_df.to_csv(output_path, encoding="utf-8-sig", index=False)

    print(f"Extraction complete. {len(result_df)} AOS triplets from {len(df)} reviews.")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    run(parse_args())
