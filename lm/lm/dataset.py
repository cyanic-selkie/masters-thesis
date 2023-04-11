from datasets import load_dataset, DatasetDict
import multiprocessing
from enum import Enum
import numpy as np
import torch

def prepare_features(examples, tokenizer, max_length, doc_stride, labels, embeddings, nodes):
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
        
    tokenized_examples["labels"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        
        answers = []

        for answer in examples["answers"][sample_index]:
            answer_start, answer_end = answer["start"], answer["end"]
            # Start token index of the current span in the text.
            token_start_index = 0
            
            while offsets[token_start_index][0] == 0 and offsets[token_start_index][1] == 0:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            
            while offsets[token_end_index][0] == 0 and offsets[token_end_index][1] == 0:
                token_end_index -= 1
            
            # Detect if the answer is out of the span.
            if (offsets[token_start_index][0] <= answer_start and offsets[token_end_index][1] >= answer_end):
                # Move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                try:
                    while offsets[token_start_index][0] < answer_start:
                        token_start_index += 1

                    while offsets[token_end_index][1] > answer_end:
                        token_end_index -= 1
                except Exception:
                    continue

                if token_start_index != token_end_index:
                    answers.append((token_start_index, token_end_index + 1))
        
        answers.sort()
        
        label_ids = [labels.index("O")] * len(input_ids)
        
        for start, end in answers:
            label_ids[start] = labels.index("B-ANSWER")
            for i in range(start + 1, end):
                label_ids[i] = labels.index("I-ANSWER")
            
        for i, token_id in enumerate(input_ids):
            if token_id in (tokenizer.cls_token_id, tokenizer.sep_token_id): #tokenizer.bos_token_id, tokenizer.eos_token_id):
                label_ids[i] = -100

        tokenized_examples["labels"].append(label_ids)
                    
    return tokenized_examples

def get_dataset(dataset, tokenizer, labels, embeddings, nodes):
    max_length = 512 # tokenizer.model_max_length
    doc_stride = max_length // 2

    train, validation = load_dataset("cyanic-selkie/wikianc-en", split=["test+validation[:80%]", "validation[-20%:]"], streaming=True)
    dataset = DatasetDict({"train": train, "validation": validation})

    dataset = dataset.remove_columns(["uuid", "article_title", "article_pageid", "article_qid", "section_heading", "section_level"])
    dataset = dataset.rename_columns({"paragraph_text": "context", "paragraph_anchors": "answers"})
    dataset = dataset.filter(lambda x: len(x["answers"]) > 0 and x["context"].strip() != "", num_proc=multiprocessing.cpu_count())
    dataset = dataset.map(lambda x: prepare_features(x, tokenizer, max_length, doc_stride, labels, embeddings, nodes), batched=True, num_proc=multiprocessing.cpu_count(), remove_columns=["context", "answers"]).filter(lambda x: labels.index("B-ANSWER") in x["labels"], num_proc=multiprocessing.cpu_count())

    dataset = dataset.shuffle(seed=42, buffer_size=1000)
    dataset.with_format(type="torch")

    return dataset
