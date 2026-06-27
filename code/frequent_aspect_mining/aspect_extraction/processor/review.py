"""
review.py

ReviewProcessor: loads a fine-tuned AOS model and extracts
Aspect-Opinion-Sentiment triplets from Korean product reviews.
"""

import logging
import math
import os
import re
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Text

import dask
import torch
from kiwipiepy import Kiwi
from transformers import AutoTokenizer

from core.model.aos import MultiInferBert
from utils.metric import find_triplet, get_spans

logger = logging.getLogger(__name__)

_PROCESSOR_DIR = Path(__file__).parent
_ROOT_DIR      = _PROCESSOR_DIR.parent

_TAIL_TAGS = {
    "JKS","JKC","JKG","JKO","JKB","JKV","JKQ","JX","JC",
    "EF","EC","SF","SP","SS","SE","SW",
}
_COPULA_RE = re.compile(r"(이에요|이었어요|이었습니다|입니다|이야|이었|이고|이라|이다)[.!?]*$")

_MAX_SEQ_LEN   = 150
_BERT_FEAT_DIM = 768
_DROPOUT       = 0.3
_N_HOPS        = 1
_CLASS_NUM     = 5


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
    """Return a SimpleNamespace with all parameters required by MultiInferBert."""
    return types.SimpleNamespace(
        model_dir        = str(_ROOT_DIR / "resources" / "saved" / "aos" / "model.pt"),
        bert_model_path  = str(_ROOT_DIR / "resources" / "saved" / "aos" / "KcELECTRA-base-v2022.pt"),
        max_sequence_len = _MAX_SEQ_LEN,
        bert_feature_dim = _BERT_FEAT_DIM,
        class_num        = _CLASS_NUM,
        dropout          = _DROPOUT,
        nhops            = _N_HOPS,
        device           = device,
    )


class ReviewProcessor:
    def __init__(self, *, device: str = "cpu", batch_size: int = 50, gpu_id: str = "0") -> None:
        if device == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        config = _model_config(device)

        self.device     = device
        self.batch_size = batch_size

        self.model = MultiInferBert(config).to(device)
        self.model.load_state_dict(torch.load(config.model_dir, map_location=device), strict=False)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(_ROOT_DIR / "resources" / "saved" / "aos")
        )
        self.kiwi = Kiwi()
        logger.info("ReviewProcessor ready.")

    def _flatten(self, lst: List[List[Any]]) -> List[Any]:
        return [item for sublist in lst for item in sublist]

    def preprocessing(self, idx: int, record: Dict[Text, Any]) -> List[Any]:
        review = record.get("corrected_text")
        if not review:
            return []
        sentences = list(dict.fromkeys(s.text for s in self.kiwi.split_into_sents(review)))
        return [[idx, s, review] for s in sentences]

    def get_tagging_instance(self, review: List[Any]) -> "TaggingInstance | None":
        review_id, sentence, origin_review = review[0], review[1], review[2]
        tokens  = sentence.strip().split()
        sen_len = len(tokens)

        bert_tokens         = self.tokenizer.encode(sentence, max_length=_MAX_SEQ_LEN)
        bert_tokens_padding = torch.zeros(_MAX_SEQ_LEN).long()
        mask                = torch.zeros(_MAX_SEQ_LEN)

        for i, tok in enumerate(bert_tokens):
            bert_tokens_padding[i] = tok
        mask[: len(bert_tokens)] = 1

        token_range = []
        token_start = 1
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

        all_ids            = []
        all_preds          = []
        all_sen_lengths    = []
        all_token_ranges   = []
        all_tokens_raw     = []
        all_probs          = []
        all_sentences      = []
        all_origin_reviews = []

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

                batch_origin = [
                    id_to_inst[sid].origin_review if sid in id_to_inst else ""
                    for sid in sentence_ids
                ]
                all_origin_reviews.extend(batch_origin)

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
                raw_aspect = self.tokenizer.decode(tokens_raw[a_start : a_end + 1])
                aspect_pos = [[t.form, str(t.tag)] for t in self.kiwi.tokenize(raw_aspect)]

                aos_set.append({
                    "sid":                  sid,
                    "preprocessed_content": origin_review,
                    "sentence":             sentence,
                    "raw_aspect":           raw_aspect,
                    "aspect_pos":           aspect_pos,
                })
        return aos_set

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

    def _extract_aspect(self, records: List[Dict[Text, Any]]) -> List[Dict[Text, Any]]:
        for record in records:
            pos = record.get("aspect_pos", [])
            raw = record.get("raw_aspect", "")

            # Strip trailing functional morphemes from raw_aspect surface text.
            # Handles particles (은/는/이/가/도/에...) and endings from the end of the span.
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
