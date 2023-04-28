from datasets import load_dataset, IterableDatasetDict, DatasetDict
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

def get_span_index(x: int, y: int, length):
    idx = ((length - x) * (length - x + 1)) // 2
    idx = -(idx - (y - x))
    return idx 

@dataclass
class DataCollatorForNELMapper(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    embedding_size: int
    devices: int
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, batch_features):
        # Tokenize just the input features (i.e., the input text).
        input_features = [{k: v for k, v in features.items() if k != "targets" and k != "spans" and k != "labels"} for features in batch_features]
        batch = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        sequence_length = batch["input_ids"].shape[1]

        # Generate the indices of the spans the model will output.
        span_indices = torch.linspace(0, sequence_length - 1, sequence_length, dtype=torch.int32)
        span_indices = torch.combinations(span_indices, 2, with_replacement=True)
        # Generate the mask the model will use to calculate the losses.
        batch_span_mask = torch.zeros(batch["input_ids"].shape[0], span_indices.shape[0])

        batch_spans = [feature["spans"] for feature in batch_features]

        batch_sparse_targets = [features["targets"] for features in batch_features]
        batch_dense_targets = torch.zeros(batch["input_ids"].shape[0], span_indices.shape[0], self.embedding_size)

        for spans, sparse_targets, span_mask, dense_targets in zip(batch_spans, batch_sparse_targets, batch_span_mask, batch_dense_targets):
            for (x, y), target in zip(spans, sparse_targets):
                idx = get_span_index(x, y, sequence_length)
                span_mask[idx] = 1
                dense_targets[idx] = target

        batch["targets"] = batch_dense_targets

        batch["span_indices"] = np.repeat(span_indices[np.newaxis, :, :], self.devices, axis=0)
        batch["span_mask"] = batch_span_mask

        return batch

@dataclass
class DataCollatorForNELClassifier(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    class_weights: torch.Tensor
    devices: int
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, batch_features):
        # Tokenize just the input features (i.e., the input text).
        batch_input_features = [{k: v for k, v in features.items() if k != "targets" and k != "spans" and k != "labels"} for features in batch_features]
        batch = self.tokenizer.pad(
            batch_input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        sequence_length = batch["input_ids"].shape[1]

        # Generate the indices of the spans the model will output.
        span_indices = torch.linspace(0, sequence_length - 1, sequence_length, dtype=torch.int32)
        span_indices = torch.combinations(span_indices, 2, with_replacement=True)
        # Generate the mask the model will use to calculate the losses.
        batch_span_mask = torch.ones(batch["input_ids"].shape[0], span_indices.shape[0])

        for input_features, span_mask in zip(batch_input_features, batch_span_mask):
            instance_sequence_length = input_features["input_ids"].shape[0]

            for x in range(instance_sequence_length, sequence_length):
                for y in range(x, sequence_length):
                    idx = get_span_index(x, y, sequence_length)
                    span_mask[idx] = 0

        batch_spans = [feature["spans"] for feature in batch_features]

        batch_sparse_labels = [features["labels"] for features in batch_features]
        batch_dense_labels = torch.zeros(batch["input_ids"].shape[0], span_indices.shape[0], dtype=torch.long)

        for spans, sparse_labels, dense_labels in zip(batch_spans, batch_sparse_labels, batch_dense_labels):
            for (x, y), label in zip(spans, sparse_labels):
                idx = get_span_index(x, y, sequence_length)
                dense_labels[idx] = label

        batch["labels"] = batch_dense_labels

        batch["span_indices"] = np.repeat(span_indices[np.newaxis, :, :], self.devices, axis=0)
        batch["class_weights"] = np.repeat(self.class_weights[np.newaxis, :], self.devices, axis=0)
        batch["span_mask"] = batch_span_mask

        return batch

def prepare_features_wikianc(examples, tokenizer, max_length, doc_stride, embeddings, nodes):
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

                # if token_start_index != token_end_index:
                spans.append((token_start_index, token_end_index + 1))
                targets.append(torch.tensor(embeddings[nodes[qid]]))
        
        if len(spans) > 0:
            spans, targets = zip(*sorted(zip(spans, targets), key=lambda x: x[0]))

        tokenized_examples["spans"].append(list(spans))
        tokenized_examples["targets"].append(list(targets))
                    
    return tokenized_examples

def prepare_features_conll(examples, tokenizer, max_length, doc_stride):
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
        
    tokenized_examples["entities"] = []
    tokenized_examples["labels"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        
        spans = []
        labels = []

        for span in examples["entities"][sample_index]:
            span_start, span_end = span["start"], span["end"]

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

                # if token_start_index != token_end_index:
                spans.append((token_start_index, token_end_index + 1))
                labels.append(1)
        
        if len(spans) > 0:
            spans, targets = zip(*sorted(zip(spans, labels), key=lambda x: x[0]))

        tokenized_examples["entities"].append(list(spans))
        tokenized_examples["labels"].append(list(labels))
                    
    return tokenized_examples

def has_spans(example, nodes):
    if example["context"].strip() == "":
        return False

    for span in example["anchors"]:
        if span["qid"] is not None and span["qid"] in nodes:
            return True

    return False

def get_dataset_wikianc(tokenizer, embedding_size, embeddings, nodes):
    nodes = {qid: i for i, qid in enumerate(pq.read_table(nodes, columns=["qid"])["qid"].to_pylist())}
    embeddings = np.memmap(embeddings, np.float32, "r", shape=(len(nodes), embedding_size))

    max_length = tokenizer.model_max_length
    doc_stride = max_length // 2

    train = load_dataset("cyanic-selkie/wikianc-en", split="train", streaming=True)
    validation = load_dataset("cyanic-selkie/wikianc-en", split="validation", streaming=True)

    dataset = IterableDatasetDict({"train": train, "validation": validation})
    
    dataset = dataset.remove_columns(["uuid", "article_title", "article_pageid", "article_qid", "section_heading", "section_level"])
    dataset = dataset.rename_columns({"paragraph_text": "context", "paragraph_anchors": "anchors"})
    dataset = dataset.filter(lambda x: has_spans(x, nodes))
    dataset = dataset.map(lambda x: prepare_features_wikianc(x, tokenizer, max_length, doc_stride, embeddings, nodes), batched=True, remove_columns=["context", "anchors"])

    dataset = dataset.shuffle(seed=42, buffer_size=1000)
    dataset = dataset.with_format(type="torch")

    return dataset

def get_dataset_conll(tokenizer):
    max_length = tokenizer.model_max_length
    doc_stride = max_length // 2

    dataset = load_dataset("cyanic-selkie/aida-conll-yago-wikidata")
    dataset = dataset.remove_columns(["uuid", "document_id", "sentence_index"])
    dataset = dataset.rename_columns({"text": "context"})
    dataset = dataset.filter(lambda x: x["context"].strip() != "" and len(x["entities"]) > 0)
    dataset = dataset.map(lambda x: prepare_features_conll(x, tokenizer, max_length, doc_stride), batched=True, remove_columns=["context", "entities"])

    dataset = dataset.shuffle(seed=42)
    dataset.set_format(type="torch")

    return dataset
