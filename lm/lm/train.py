import os
import argparse
from dataset import get_dataset, DataCollatorForNEL
from model import instantiate_model
from transformers import TrainingArguments, Trainer
import torch
from torch import nn
import numpy as np
import evaluate

def train(model, tokenizer, name, batch_size, learning_rate, warmup_steps, epochs, gradient_accumulation_steps, embedding_size, dataset):
    data_collator = DataCollatorForNEL(tokenizer, embedding_size)

    args = TrainingArguments(
        output_dir=f"models/{name}",
        evaluation_strategy="no",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        report_to="clearml",
        save_strategy="steps",
        save_steps=500,
        save_total_limit=5,
        logging_steps=20,
        warmup_steps=warmup_steps
        gradient_accumulation_steps=gradient_accumulation_steps
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(f"models/{name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--embedding-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--warmup-steps", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--nodes", type=str, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    args = parser.parse_args()

    embeddings, nodes = None, None

    os.environ["CLEARML_PROJECT"] = "masters-thesis"
    os.environ["CLEARML_TASK"] = args.name

    model, tokenizer = instantiate_model(args.checkpoint, 128, 0.8)

    dataset = get_dataset(tokenizer, embeddings, nodes)

    train(model, tokenizer, args.name, args.batch_size, args.learning_rate, args.warmup_steps, args.epochs, args.gradient_accumulation_steps, args.embedding_size, dataset)