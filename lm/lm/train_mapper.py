import os
import argparse
from dataset import get_dataset_wikianc, DataCollatorForNELMapper
from model import instantiate_model
from transformers import TrainingArguments, Trainer
import torch
from torch import nn
import numpy as np
import evaluate
import os
import pyarrow.parquet as pq
from math import ceil

def train(model, checkpoint, tokenizer, name, batch_size, learning_rate, warmup_steps, gradient_accumulation_steps, devices, embedding_size, classes_num, dataset, continue_training):
    data_collator = DataCollatorForNELMapper(tokenizer, embedding_size, devices)

    args = TrainingArguments(
        output_dir=f"models/{name}",
        evaluation_strategy="no",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=1,
        weight_decay=0.01,
        report_to="wandb",
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=20,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=ceil(34555183 / (gradient_accumulation_steps * batch_size)),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    if continue_training:
        trainer.train(checkpoint)
    else:
        trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--classes-num", type=int, required=True)
    parser.add_argument("--embedding-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--warmup-steps", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--devices", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--nodes", type=str, required=True)
    parser.add_argument("--continue-training", action="store_true")
    args = parser.parse_args()

    model, tokenizer = instantiate_model(args.checkpoint, args.embedding_size, args.classes_num)

    dataset = get_dataset_wikianc(tokenizer, args.embedding_size, args.embeddings, args.nodes)

    train(model, args.checkpoint, tokenizer, args.name, args.batch_size, args.learning_rate, args.warmup_steps, args.gradient_accumulation_steps, args.devices, args.embedding_size, args.classes_num, dataset, args.continue_training)