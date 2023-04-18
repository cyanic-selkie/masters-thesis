from datasets import load_dataset, IterableDatasetDict
from transformers.data.data_collator import DataCollatorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
import multiprocessing
from enum import Enum
import numpy as np
import torch
from typing import Union, Optional
from dataclasses import dataclass
import pyarrow.parquet as pq

@dataclass
class DataCollatorForNEL(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    embedding_size: int
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        batch_targets = [feature["targets"] for feature in features]
        batch_spans = [feature["spans"] for feature in features]

        no_labels_features = [{k: v for k, v in feature.items() if k != "targets" and k != "spans"} for feature in features]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        span_indices = torch.linspace(0, sequence_length - 1, sequence_length, dtype=torch.int32)
        span_indices = torch.combinations(span_indices, 2, with_replacement=True)

        span_mask = torch.zeros(len(batch_targets), span_indices.shape[0])
        batch_dense_targets = torch.zeros(len(batch_targets), span_indices.shape[0], self.embedding_size)

        for spans, targets, mask, dense_targets in zip(batch_spans, batch_targets, span_mask, batch_dense_targets):
            for (x, y), target in zip(spans, targets):
                idx = ((sequence_length - x) * (sequence_length - x + 1)) // 2
                idx = -(idx - (y - x))
                mask[idx] = 1
                dense_targets[idx] = target

        batch["spans"] = span_indices
        batch["span_mask"] = span_mask
        batch["targets"] = batch_dense_targets

        return batch

def prepare_features(examples, tokenizer, max_length, doc_stride, embeddings, nodes):
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

            if qid is None:
                continue
            # Start token index of the current span in the text.
            token_start_index = 0
            
            while offsets[token_start_index][0] == 0 and offsets[token_start_index][1] == 0:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            
            while offsets[token_end_index][0] == 0 and offsets[token_end_index][1] == 0:
                token_end_index -= 1
            
            # Detect if the span is out of the sequence length.
            if (offsets[token_start_index][0] <= span_start and offsets[token_end_index][1] >= span_end):
                # Move the token_start_index and token_end_index to the two ends of the span.
                # Note: we could go after the last offset if the span is the last word (edge case).
                try:
                    while offsets[token_start_index][0] < span_start:
                        token_start_index += 1

                    while offsets[token_end_index][1] > span_end:
                        token_end_index -= 1
                except Exception:
                    continue

                if token_start_index != token_end_index:
                    spans.append((token_start_index, token_end_index + 1))
                    targets.append(embeddings[nodes[qid]])
        
        spans, targets = zip(*sorted(zip(spans, targets), key=lambda x: x[0]))

        tokenized_examples["spans"].append(list(spans))
        tokenized_examples["targets"].append(list(targets))
                    
    return tokenized_examples

def get_dataset(tokenizer, embedding_size, embeddings, nodes):
    nodes = {qid: i for i, qid in enumerate(pq.read_table(nodes, columns=["qid"])["qid"].to_pylist())}
    embeddings = np.memmap(embeddings, np.float32, "r", shape=(len(nodes), embedding_size))

    max_length = tokenizer.model_max_length
    doc_stride = max_length // 2

    train = load_dataset("cyanic-selkie/wikianc-en", split="train", streaming=True)
    validation = load_dataset("cyanic-selkie/wikianc-en", split="validation", streaming=True)

    dataset = IterableDatasetDict({"train": train, "validation": validation})
    
    dataset = dataset.remove_columns(["uuid", "article_title", "article_pageid", "article_qid", "section_heading", "section_level"])
    dataset = dataset.rename_columns({"paragraph_text": "context", "paragraph_anchors": "anchors"})
    dataset = dataset.filter(lambda x: len(x["anchors"]) > 0 and x["context"].strip() != "" and any(y["qid"] is not None for y in x["anchors"]))
    dataset = dataset.map(lambda x: prepare_features(x, tokenizer, max_length, doc_stride, embeddings, nodes), batched=True, remove_columns=["context", "anchors"])

    dataset = dataset.shuffle(seed=42, buffer_size=1000)
    dataset = dataset.with_format(type="torch")

    return dataset
