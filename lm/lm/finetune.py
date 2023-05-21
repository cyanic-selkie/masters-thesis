import os
import argparse
from dataset import get_dataset_conll, DataCollatorForEL
from model import instantiate_model
from transformers import TrainingArguments, Trainer, TrainerCallback
import torch
from torch import nn
import numpy as np
import os
from math import ceil

def train(model, checkpoint, tokenizer, name, batch_size, learning_rate, warmup_steps, gradient_accumulation_steps, embedding_size, dataset, continue_training, epochs):
    data_collator = DataCollatorForEL(tokenizer, embedding_size)

    args = TrainingArguments(
        output_dir=f"models/{name}",
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        report_to="wandb",
        save_strategy="epoch",
        save_total_limit=epochs,
        logging_steps=1,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.evaluate()
    return

    if continue_training:
        trainer.train(checkpoint)
    else:
        trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--embedding-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--warmup-steps", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--nodes", type=str, required=True)
    parser.add_argument("--continue-training", action="store_true")
    args = parser.parse_args()

    model, tokenizer = instantiate_model(args.checkpoint, args.embedding_size)

    dataset = get_dataset_conll(tokenizer, args.embedding_size, args.embeddings, args.nodes)

    train(model, args.checkpoint, tokenizer, args.name, args.batch_size, args.learning_rate, args.warmup_steps, args.gradient_accumulation_steps, args.embedding_size, dataset, args.continue_training, args.epochs)