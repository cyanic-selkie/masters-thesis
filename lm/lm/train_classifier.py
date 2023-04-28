import os
from math import comb
import argparse
from dataset import get_dataset_conll, DataCollatorForNELClassifier
from model import instantiate_model
from transformers import TrainingArguments, Trainer
import torch
from torch import nn
import numpy as np
import evaluate
import clearml
from clearml import Task
import os
import pyarrow.parquet as pq
import evaluate

def compute_metrics(p, seqeval, id2label):
    predictions, labels = p
    predictions = np.argmax(predictions[1], axis=2)

    true_predictions = [
        [id2label[p] for p in prediction]
        for prediction in predictions
    ]
    true_labels = [
        [id2label[l] for l in label]
        for label in labels
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def train(model, checkpoint, tokenizer, name, batch_size, learning_rate, warmup_steps, epochs, gradient_accumulation_steps, devices, embedding_size, classes_num, dataset):
    data_collator = DataCollatorForNELClassifier(tokenizer, weights, devices)
    seqeval = evaluate.load("seqeval")

    args = TrainingArguments(
        output_dir=f"models/{name}",
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        report_to="none",
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=1,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, seqeval, model.config.id2label)
    )

    # if os.path.isdir(checkpoint):
        # trainer.train(checkpoint)
    # else:
        # trainer.train()
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--embedding-size", type=int, required=True)
    parser.add_argument("--classes-num", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--warmup-steps", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--devices", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    args = parser.parse_args()

    model, tokenizer = instantiate_model(args.checkpoint, args.embedding_size, args.classes_num)

    dataset = get_dataset_conll(tokenizer)

    frequencies = torch.zeros(args.classes_num)
    for example in dataset["train"]:
        sequence_length = example["input_ids"].shape[0]

        total = sequence_length + comb(sequence_length, 2)

        if len(example["labels"]) == 0:
            frequencies[0] += total
        else:
            counts = torch.bincount(example["labels"])
            frequencies[0] += total - counts.sum()
            for i in range(1, args.classes_num):
                frequencies[i] += counts[i]
    frequencies /= frequencies.sum()

    for name, param in model.named_parameters():
        if not "classifier" in name:
            param.requires_grad = False

    weights = 1 / frequencies
    data_collator = DataCollatorForNELClassifier(tokenizer, weights, args.devices)

    x = iter(dataset["train"])
    dataset["train"] = [next(x) for _ in range(4)]

    dataset["validation"] = dataset["train"]

    train(model, args.checkpoint, tokenizer, args.name, args.batch_size, args.learning_rate, args.warmup_steps, args.epochs, args.gradient_accumulation_steps, args.devices, args.embedding_size, weights, dataset)