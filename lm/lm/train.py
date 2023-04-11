import os
import argparse
from dataset import get_dataset
from model import instantiate_model
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
import torch
from torch import nn
import numpy as np
import evaluate

def compute_metrics(p, metric, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def train(model, tokenizer, config, name, batch_size, learning_rate, warmup_ratio, epochs, dataset, metric):
	data_collator = DataCollatorForTokenClassification(tokenizer)

	args = TrainingArguments(
		output_dir=f"models/{name}",
	    evaluation_strategy="epoch",
	    learning_rate=learning_rate,
	    per_device_train_batch_size=batch_size,
	    per_device_eval_batch_size=batch_size,
	    num_train_epochs=epochs,
	    weight_decay=0.01,
	    report_to="clearml",
	    save_strategy="epoch",
	    save_total_limit=1,
	    load_best_model_at_end=True,
	    logging_steps=20,
		warmup_ratio=warmup_ratio
	)

	trainer = Trainer(
	    model,
	    args,
	    train_dataset=dataset["train"],
	    eval_dataset=dataset["validation"],
	    data_collator=data_collator,
	    tokenizer=tokenizer,
	    compute_metrics=lambda x: compute_metrics(x, metric, labels),
	)

	trainer.train()
	trainer.save_model(f"models/{name}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--checkpoint", type=str, required=True)
	parser.add_argument("--frozen", action="store_true")
	parser.add_argument("--dataset", choices=["wikipedia", "squad"], required=True)
	parser.add_argument("--name", type=str, required=True)
	parser.add_argument("--learning-rate", type=float)
	parser.add_argument("--warmup-ratio", type=float)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--epochs", type=int)
	parser.add_argument("--embeddings", type=str, required=True)
	parser.add_argument("--nodes", type=str, required=True)
	args = parser.parse_args()

	embeddings, nodes = None, None

	os.environ["CLEARML_PROJECT"] = "answer-detection"
	os.environ["CLEARML_TASK"] = args.name

	labels = ["O", "B-ANSWER", "I-ANSWER"]

	model, tokenizer, config = instantiate_model(args.checkpoint, "roberta", labels, args.frozen, 128, 0.8)

	breakpoint()

	learning_rate = args.learning_rate or (1e-4 if args.frozen else 1e-6)
	warmup_ratio = args.warmup_ratio or (0 if args.frozen else 0.33)
	epochs = args.epochs or (3 if args.frozen else 6)

	dataset = get_dataset(args.dataset, tokenizer, labels, embeddings, nodes)

	train(model, tokenizer, config, args.name, args.batch_size, learning_rate, warmup_ratio, epochs, dataset, evaluate.load("seqeval"))