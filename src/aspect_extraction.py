"""
aspect_extraction.py

Extracts Aspect-Opinion-Sentiment (AOS) triplets from Korean product reviews
using a fine-tuned KcELECTRA-based model.

Inference:
    python -m src.aspect_extraction [--input PATH] [--output PATH] [--batch-size N]

Fine-tuning:
    Called via main_fine-tuning.py, or use parse_finetune_args() / finetune() directly.
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

import torch
import torch.nn
import torch.nn.functional as F
import dask
import pandas as pd
from kiwipiepy import Kiwi
from sklearn.metrics import classification_report, confusion_matrix  # fine-tuning only
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertTokenizer  # BertTokenizer: fine-tuning only

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
# Model architecture — shared by inference and fine-tuning
# ---------------------------------------------------------------------------

class MultiInferBert(torch.nn.Module):
    def __init__(self, args, pretrained=False):
        super().__init__()
        self.args = args
        if pretrained:
            self.bert = AutoModel.from_pretrained(args.bert_model_path)
        else:
            self.bert = AutoModel.from_config(AutoConfig.from_pretrained(args.bert_model_path))
        self.cls_linear     = torch.nn.Linear(args.bert_feature_dim * 2, args.class_num)
        self.feature_linear = torch.nn.Linear(
            args.bert_feature_dim * 2 + args.class_num * 3, args.bert_feature_dim * 2
        )
        self.dropout_output = torch.nn.Dropout(p=args.dropout)

    def multi_hops(self, features, mask, k):
        """Iteratively refine grid logits via k rounds of multi-hop inference."""
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
# Inference — pipeline step
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


# ---------------------------------------------------------------------------
# Fine-tuning — data pipeline
# ---------------------------------------------------------------------------

_SENTIMENT2ID = {"negative": 3, "positive": 4}

_DEFAULT_TRAIN_DATA = _ROOT / "data" / "kc_electra_train_example.json"
_DEFAULT_DEV_DATA   = _ROOT / "data" / "kc_electra_dev_example.json"
_FINETUNE_MODEL_DIR = _ROOT / "checkpoints" / "kc_electra"


def _bio_spans(tags_str: str):
    """Parse a space-separated BIO tag string into word-index [start, end] spans."""
    tags  = tags_str.strip().split()
    spans = []
    start = -1
    for i, t in enumerate(tags):
        if t.endswith("B"):
            if start != -1:
                spans.append([start, i - 1])
            start = i
        else:
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, len(tags) - 1])
    return spans


class _TrainingInstance:
    def __init__(self, tokenizer, sentence_pack, args):
        self.id       = sentence_pack["id"]
        self.sentence = sentence_pack["triples"][0]["sentence"]
        self.tokens   = self.sentence.strip().split()
        self.sen_length  = len(self.tokens)
        self.token_range = []

        self.bert_tokens         = tokenizer.encode(self.sentence, max_length=args.max_sequence_len)
        self.length              = len(self.bert_tokens)
        self.bert_tokens_padding = torch.zeros(args.max_sequence_len).long()
        self.tags                = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        self.mask                = torch.zeros(args.max_sequence_len)

        for i in range(self.length):
            self.bert_tokens_padding[i] = self.bert_tokens[i]
        self.mask[: self.length] = 1

        token_start = 1
        for w in self.tokens:
            token_end = token_start + len(tokenizer.encode(w, add_special_tokens=False))
            self.token_range.append([token_start, token_end - 1])
            token_start = token_end

        assert self.length == self.token_range[-1][-1] + 2

        self.tags[:, :] = -1
        for i in range(1, self.length - 1):
            for j in range(i, self.length - 1):
                self.tags[i][j] = 0

        for triple in sentence_pack["triples"]:
            aspect_span  = _bio_spans(triple["target_tags"])
            opinion_span = _bio_spans(triple["opinion_tags"])

            for l, r in aspect_span:
                start = self.token_range[l][0]
                end   = self.token_range[r][1]
                for i in range(start, end + 1):
                    for j in range(i, end + 1):
                        self.tags[i][j] = 1
                for i in range(l, r + 1):
                    al, ar = self.token_range[i]
                    self.tags[al + 1 : ar + 1, :] = -1
                    self.tags[:, al + 1 : ar + 1] = -1

            for l, r in opinion_span:
                start = self.token_range[l][0]
                end   = self.token_range[r][1]
                for i in range(start, end + 1):
                    for j in range(i, end + 1):
                        self.tags[i][j] = 2
                for i in range(l, r + 1):
                    pl, pr = self.token_range[i]
                    self.tags[pl + 1 : pr + 1, :] = -1
                    self.tags[:, pl + 1 : pr + 1] = -1

            for al, ar in aspect_span:
                for pl, pr in opinion_span:
                    for i in range(al, ar + 1):
                        for j in range(pl, pr + 1):
                            sal, sar = self.token_range[i]
                            spl, spr = self.token_range[j]
                            self.tags[sal : sar + 1, spl : spr + 1] = -1
                            if i > j:
                                self.tags[spl][sal] = _SENTIMENT2ID[triple["sentiment"]]
                            else:
                                self.tags[sal][spl] = _SENTIMENT2ID[triple["sentiment"]]


def _load_training_instances(sentence_packs, args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_path)
    instances = []
    for pack in sentence_packs:
        try:
            instances.append(_TrainingInstance(tokenizer, pack, args))
        except Exception:
            continue
    return instances


class _TrainingDataIterator:
    def __init__(self, instances, args):
        self.instances   = instances
        self.args        = args
        self.batch_count = math.ceil(len(instances) / args.batch_size)

    def get_batch(self, index):
        lo = index * self.args.batch_size
        hi = min(lo + self.args.batch_size, len(self.instances))
        batch = self.instances[lo:hi]

        return (
            [inst.id          for inst in batch],
            torch.stack([inst.bert_tokens_padding for inst in batch]).to(self.args.device),
            torch.tensor([inst.length             for inst in batch]).to(self.args.device),
            torch.stack([inst.mask                for inst in batch]).to(self.args.device),
            [inst.sen_length  for inst in batch],
            [inst.token_range for inst in batch],
            torch.stack([inst.tags                for inst in batch]).to(self.args.device),
            [inst.bert_tokens for inst in batch],
            [inst.sentence    for inst in batch],
        )


# ---------------------------------------------------------------------------
# Fine-tuning — engine
# ---------------------------------------------------------------------------

class _TrainingEngine:
    def __init__(self, args):
        self.model = MultiInferBert(args, pretrained=True).to(args.device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.bert.parameters(),       "lr": args.learning_rate},
                {"params": self.model.cls_linear.parameters()},
            ],
            lr=5e-5,
        )
        self.clip = 5

    def train_step(self, tokens, masks, tags):
        self.model.train()
        self.optimizer.zero_grad()
        preds         = self.model(tokens, masks)
        preds_flatten = preds.reshape([-1, preds.shape[3]])
        tags_flatten  = tags.reshape([-1])
        loss          = F.cross_entropy(preds_flatten, tags_flatten, ignore_index=-1)
        loss.backward()
        if self.clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        return loss.item()

    def eval(self, dataset):
        self.model.eval()
        with torch.no_grad():
            all_preds, all_labels, all_lengths = [], [], []
            all_sens_lengths, all_token_ranges = [], []
            bert_tokens_real = sentences = None

            for i in range(dataset.batch_count):
                _, bert_tokens, lengths, masks, sens_lens, token_ranges, tags, bert_tokens_real, sentences = dataset.get_batch(i)
                preds = torch.argmax(self.model(bert_tokens, masks), dim=3)
                all_preds.append(preds)
                all_labels.append(tags)
                all_lengths.append(lengths)
                all_sens_lengths.extend(sens_lens)
                all_token_ranges.extend(token_ranges)

            all_preds   = torch.cat(all_preds,   dim=0).cpu().tolist()
            all_labels  = torch.cat(all_labels,  dim=0).cpu().tolist()
            all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()

        metric = _TrainingMetric(
            all_preds, all_labels, all_lengths,
            all_sens_lengths, all_token_ranges,
            bert_tokens_real, sentences,
        )
        p, r, f1, report, confu, apc_f1 = metric.score_union_tags()
        return p, r, f1, report, confu, apc_f1, metric.score_aspect(), metric.score_opinion()


# ---------------------------------------------------------------------------
# Fine-tuning — metrics
# ---------------------------------------------------------------------------

class _TrainingMetric:
    def __init__(self, predictions, goldens, bert_lengths, sen_lengths,
                 tokens_ranges, bert_tokens_real, sentences, ignore_index=-1):
        self.predictions      = predictions
        self.goldens          = goldens
        self.sen_lengths      = sen_lengths
        self.tokens_ranges    = tokens_ranges
        self.ignore_index     = ignore_index
        self.data_num         = len(predictions)

    def _spans(self, tags, length, token_range, span_type):
        spans = []
        start = -1
        for i in range(length):
            l, _ = token_range[i]
            if tags[l][l] == self.ignore_index:
                continue
            elif tags[l][l] == span_type:
                if start == -1:
                    start = i
            elif start != -1:
                spans.append([start, i - 1])
                start = -1
        if start != -1:
            spans.append([start, length - 1])
        return spans

    def _triplets(self, tags, aspect_spans, opinion_spans, token_ranges):
        result = []
        for al, ar in aspect_spans:
            for pl, pr in opinion_spans:
                tag_num = [0] * 5
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        a0 = token_ranges[i][0]
                        o0 = token_ranges[j][0]
                        tag_num[int(tags[a0][o0] if al < pl else tags[o0][a0])] += 1
                if sum(tag_num[3:]) == 0:
                    continue
                result.append([al, ar, pl, pr, 5 if tag_num[4] >= tag_num[3] else 3])
        return result

    def _prf(self, golden_set, predicted_set):
        correct   = len(golden_set & predicted_set)
        precision = correct / len(predicted_set) if predicted_set else 0
        recall    = correct / len(golden_set)    if golden_set    else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        return precision, recall, f1

    def score_aspect(self):
        g, p = set(), set()
        for i in range(self.data_num):
            for s in self._spans(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i], 1):
                g.add(f"{i}-{'-'.join(map(str, s))}")
            for s in self._spans(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i], 1):
                p.add(f"{i}-{'-'.join(map(str, s))}")
        return self._prf(g, p)

    def score_opinion(self):
        g, p = set(), set()
        for i in range(self.data_num):
            for s in self._spans(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i], 2):
                g.add(f"{i}-{'-'.join(map(str, s))}")
            for s in self._spans(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i], 2):
                p.add(f"{i}-{'-'.join(map(str, s))}")
        return self._prf(g, p)

    def score_union_tags(self):
        g_set, p_set = set(), set()
        g_lst, p_lst = [], []
        for i in range(self.data_num):
            ga = self._spans(self.goldens[i],     self.sen_lengths[i], self.tokens_ranges[i], 1)
            go = self._spans(self.goldens[i],     self.sen_lengths[i], self.tokens_ranges[i], 2)
            pa = self._spans(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i], 1)
            po = self._spans(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i], 2)
            gt = self._triplets(self.goldens[i],     ga, go, self.tokens_ranges[i])
            pt = self._triplets(self.predictions[i], pa, po, self.tokens_ranges[i])
            for t in gt:
                g_set.add(f"{i}-{'-'.join(map(str, t))}")
            for t in pt:
                p_set.add(f"{i}-{'-'.join(map(str, t))}")
            for r_i in range(len(gt)):
                g_lst.append(gt[r_i][-1])
                p_lst.append(pt[r_i][-1] if r_i < len(pt) else -1)

        p, r, f1 = self._prf(g_set, p_set)
        report = classification_report(
            g_lst, p_lst, labels=[3, 5],
            target_names=["negative", "positive"], digits=2, zero_division=0,
        )
        confu  = confusion_matrix(g_lst, p_lst)
        apc_f1 = float(report.split()[-14]) if report.split() else 0.0
        return p, r, f1, report, confu, apc_f1


# ---------------------------------------------------------------------------
# Fine-tuning — entry point
# ---------------------------------------------------------------------------

def parse_finetune_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune KcELECTRA for AOS triplet extraction."
    )
    parser.add_argument("--train_data",          type=str,   default=str(_DEFAULT_TRAIN_DATA))
    parser.add_argument("--dev_data",            type=str,   default=str(_DEFAULT_DEV_DATA))
    parser.add_argument("--model_dir",           type=str,   default=str(_FINETUNE_MODEL_DIR))
    parser.add_argument("--max_sequence_len",    type=int,   default=_MAX_SEQ_LEN)
    parser.add_argument("--device",              type=str,   default="cuda")
    parser.add_argument("--bert_model_path",     type=str,   default=str(_CHECKPOINT_DIR))
    parser.add_argument("--bert_tokenizer_path", type=str,   default=str(_CHECKPOINT_DIR))
    parser.add_argument("--bert_feature_dim",    type=int,   default=_BERT_FEAT_DIM)
    parser.add_argument("--learning_rate",       type=float, default=5e-5)
    parser.add_argument("--dropout",             type=float, default=_DROPOUT)
    parser.add_argument("--nhops",               type=int,   default=_N_HOPS)
    parser.add_argument("--batch_size",          type=int,   default=16)
    parser.add_argument("--epochs",              type=int,   default=25)
    parser.add_argument("--class_num",           type=int,   default=_CLASS_NUM)
    return parser.parse_args(args)


def finetune(args: argparse.Namespace) -> None:
    import json
    import random
    import time
    import numpy as np
    from tqdm import trange

    trainer = _TrainingEngine(args)

    train_packs = json.load(open(args.train_data))
    dev_packs   = json.load(open(args.dev_data))
    random.shuffle(train_packs)

    instances_train = _load_training_instances(train_packs, args)
    instances_dev   = _load_training_instances(dev_packs,   args)
    print(f"Train: {len(instances_train)}  Dev: {len(instances_dev)}")
    random.shuffle(instances_train)

    trainset = _TrainingDataIterator(instances_train, args)
    devset   = _TrainingDataIterator(instances_dev,   args)

    os.makedirs(args.model_dir, exist_ok=True)
    timestamp = time.strftime("%y%m%d_%H%M%S")

    best_f1, best_epoch, train_loss = 0, 0, []

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        for j in trange(trainset.batch_count):
            _, bert_tokens, _, masks, _, _, tags, _, _ = trainset.get_batch(j)
            train_loss.append(trainer.train_step(bert_tokens, masks, tags))

        p, r, f1, report, confu, apc_f1, aspect_res, opinion_res = trainer.eval(devset)
        print(confu)
        print(report)
        print("Aspect\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}".format(*aspect_res))
        print("Opinion\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}".format(*opinion_res))
        print("Triplet\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}\tAPC_F1:{:.5f}\n".format(p, r, f1, apc_f1))

        if f1 > best_f1:
            torch.save(
                trainer.model,
                os.path.join(args.model_dir, f"model_kc_electra_{timestamp}.pt"),
            )
            best_f1, best_epoch = f1, epoch

        print(f"epoch:{epoch}  loss:{np.mean(train_loss):.4f}")

    print(f"Best epoch: {best_epoch}  Best dev F1: {best_f1:.5f}")


if __name__ == "__main__":
    run(parse_args())
