from datasets import load_dataset, IterableDatasetDict
from transformers.data.data_collator import DataCollatorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
import numpy as np
import torch
from typing import Union, Optional
from dataclasses import dataclass
import pyarrow.parquet as pq
import random


def bernoulli(probability: float) -> bool:
    return random.random() < probability


@dataclass
class DataCollatorForEL(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    embedding_size: int
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, batch_features):
        # Tokenize just the input features (i.e., the input text).
        input_features = [{
            k: v
            for k, v in features.items()
            if k != "targets" and k != "spans" and k != "labels"
        } for features in batch_features]
        batch = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        spans_max_length = max(
            len(features["spans"]) for features in batch_features)

        batch_spans = torch.stack([
            torch.cat((feature["spans"],
                       torch.zeros(spans_max_length - len(feature["spans"]),
                                   2,
                                   dtype=torch.int64)),
                      dim=0) for feature in batch_features
        ], 0)
        batch_targets = torch.stack([
            torch.cat((feature["targets"],
                       torch.zeros(spans_max_length - len(feature["spans"]),
                                   self.embedding_size,
                                   dtype=torch.int64)),
                      dim=0) for feature in batch_features
        ], 0)

        batch["spans"] = batch_spans
        batch["targets"] = batch_targets
        batch["return_loss"] = True

        return batch


def prepare_features(examples, tokenizer, max_length, doc_stride, embeddings,
                     nodes, masked):
    tokenized_examples = tokenizer(
        examples["context"],
        truncation=True,
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["spans"] = []
    tokenized_examples["targets"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]

        spans = []
        targets = []

        for span in examples["anchors"][sample_index]:
            span_start, span_end, qid = span["start"], span["end"], span["qid"]

            if qid is None or qid not in nodes:
                continue

            # Start token index of the current span in the text.
            token_start_index = 0

            while offsets[token_start_index][0] == 0 and offsets[
                    token_start_index][1] == 0:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1

            while offsets[token_end_index][0] == 0 and offsets[
                    token_end_index][1] == 0:
                token_end_index -= 1

            # Detect if the span is out of the sequence length.
            if (offsets[token_start_index][0] <= span_start
                    and offsets[token_end_index][1] >= span_end):
                # Move the token_start_index and token_end_index to the two ends of the span.
                # Note: we could go after the last offset if the span is the last word (edge case).
                try:
                    while offsets[token_start_index][0] < span_start:
                        token_start_index += 1

                    while offsets[token_end_index][1] > span_end:
                        token_end_index -= 1
                except Exception:
                    continue

                spans.append((token_start_index, token_end_index))
                targets.append(torch.tensor(embeddings[nodes[qid]]))

        if len(spans) > 0:
            spans, targets = zip(
                *sorted(zip(spans, targets), key=lambda x: x[0]))

        if masked:
            for x, y in spans:
                if bernoulli(0.8):
                    input_ids[x:y + 1] = [
                        tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
                    ] * (y + 1 - x)
                elif bernoulli(0.5):
                    input_ids[x:y + 1] = [
                        random.randint(0, tokenizer.vocab_size - 1)
                        for _ in range(y + 1 - x)
                    ]

        tokenized_examples["spans"].append(list(spans))
        tokenized_examples["targets"].append(list(targets))

    return tokenized_examples


def has_spans(example, nodes):
    if example["context"].strip() == "":
        return False

    for span in example["anchors"]:
        if span["qid"] is not None and span["qid"] in nodes:
            return True

    return False


def get_dataset_wikianc(tokenizer, embedding_size, embeddings, nodes):
    nodes = {
        qid: i
        for i, qid in enumerate(
            pq.read_table(nodes, columns=["qid"])["qid"].to_pylist())
    }
    embeddings = np.memmap(embeddings,
                           np.float32,
                           "r",
                           shape=(len(nodes), embedding_size))

    max_length = tokenizer.model_max_length
    doc_stride = max_length // 2

    train = load_dataset("cyanic-selkie/wikianc-en",
                         split="train",
                         streaming=True)
    validation = load_dataset("cyanic-selkie/wikianc-en",
                              split="validation",
                              streaming=True)

    dataset = IterableDatasetDict({"train": train, "validation": validation})

    dataset = dataset.remove_columns([
        "uuid", "article_title", "article_pageid", "article_qid",
        "section_heading", "section_level"
    ])
    dataset = dataset.rename_columns({
        "paragraph_text": "context",
        "paragraph_anchors": "anchors"
    })
    dataset = dataset.filter(lambda x: has_spans(x, nodes))
    dataset = dataset.map(lambda x: prepare_features(
        x, tokenizer, max_length, doc_stride, embeddings, nodes, True),
                          batched=True,
                          remove_columns=["context", "anchors"])
    dataset = dataset.filter(lambda x: len(x["spans"]) > 0)

    dataset = dataset.shuffle(seed=42, buffer_size=1000)
    dataset = dataset.with_format(type="torch")

    return dataset


def get_dataset_conll(tokenizer, embedding_size, embeddings, nodes):
    nodes = {
        qid: i
        for i, qid in enumerate(
            pq.read_table(nodes, columns=["qid"])["qid"].to_pylist())
    }
    embeddings = np.memmap(embeddings,
                           np.float32,
                           "r",
                           shape=(len(nodes), embedding_size))

    max_length = tokenizer.model_max_length
    doc_stride = max_length // 2

    dataset = load_dataset("cyanic-selkie/aida-conll-yago-wikidata")

    dataset = dataset.remove_columns(["uuid", "document_id"])
    dataset = dataset.rename_columns({
        "text": "context",
        "entities": "anchors"
    })
    dataset = dataset.filter(lambda x: has_spans(x, nodes))
    dataset = dataset.map(lambda x: prepare_features(
        x, tokenizer, max_length, doc_stride, embeddings, nodes, False),
                          batched=True,
                          remove_columns=["context", "anchors"])
    dataset = dataset.filter(lambda x: len(x["spans"]) > 0)

    dataset = dataset.shuffle(seed=42)
    dataset.set_format(type="torch")

    return dataset


def get_dataset_evaluate():

    dataset = load_dataset("cyanic-selkie/aida-conll-yago-wikidata")

    return dataset
