"""
Train XLM-RoBERTa-large on NER dataset using the transformers library.
Supports CONLL format input.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import Dataset, DatasetDict
import seqeval
import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
def parse_args():
    parser = argparse.ArgumentParser(description="Train XLM-RoBERTa for NER")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-large", help="Pretrained model name")
    parser.add_argument("--data_dir", type=str, default="./review", help="Directory with CONLL files")
    parser.add_argument("--output_dir", type=str, default="./xlm_roberta_ner_model", help="Directory to save model and outputs")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data to use for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def read_conll_file(file_path: str) -> List[tuple[list[str], list[str]]]:
    """Read CONLL format file and return list of (tokens, tags) tuples."""
    sentences = []
    tokens = []
    tags = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append((tokens, tags))
                    tokens = []
                    tags = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    token = parts[0]
                    tag = parts[1]
                    tokens.append(token)
                    tags.append(tag)
    
    if tokens:
        sentences.append((tokens, tags))
    
    return sentences


def get_unique_tags(data: list[tuple[list[str], list[str]]]) -> list[str]:
    """Extract unique tags from data."""
    tags = set()
    for _, tag_list in data:
        tags.update(tag_list)
    return sorted(list(tags))


def create_tag_mappings(tag_list: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Create tag to id and id to tag mappings."""
    tag2id = {tag: idx for idx, tag in enumerate(tag_list)}
    id2tag = {idx: tag for tag, idx in tag2id.items()}
    return tag2id, id2tag


def tokenize_and_align_labels(
    examples: dict,
    tokenizer,
    tag2id: dict[str, int],
    max_length: int,
) -> dict:
    """Tokenize text and align labels with subword tokens."""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
        padding="max_length",
    )
    
    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(tag2id[label[word_idx]])
            else:
                # For subword tokens, use the same label as the first subword
                label_ids.append(tag2id[label[word_idx]])
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def load_and_prepare_dataset(
    data_dir: Path,
    tokenizer,
    max_length: int,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[DatasetDict, dict[str, int], dict[int, str]]:
    """Load CONLL files and prepare HuggingFace dataset."""
    
    logger.info("Loading CONLL files...")
    all_data = []
    
    # Load all CONLL files
    for conll_file in data_dir.glob("*.conll"):
        logger.info(f"Loading {conll_file.name}...")
        data = read_conll_file(str(conll_file))
        all_data.extend(data)
        logger.info(f"  Loaded {len(data)} sentences")
    
    if not all_data:
        raise ValueError(f"No CONLL files found in {data_dir}")
    
    logger.info(f"Total sentences loaded: {len(all_data)}")
    
    # Extract unique tags
    tag_list = get_unique_tags(all_data)
    tag2id, id2tag = create_tag_mappings(tag_list)
    
    logger.info(f"Found {len(tag_list)} unique tags: {tag_list}")
    logger.info(f"Tag mappings: {tag2id}")
    
    # Split into train/eval
    np.random.seed(seed)
    indices = np.random.permutation(len(all_data))
    split_idx = int(len(all_data) * train_ratio)
    
    train_data = [all_data[i] for i in indices[:split_idx]]
    eval_data = [all_data[i] for i in indices[split_idx:]]
    
    logger.info(f"Train: {len(train_data)}, Eval: {len(eval_data)}")
    
    # Convert to HF datasets format
    def create_hf_dataset(data: list[tuple[list[str], list[str]]]):
        tokens_list = [item[0] for item in data]
        tags_list = [item[1] for item in data]
        return Dataset.from_dict({
            "tokens": tokens_list,
            "tags": tags_list,
        })
    
    train_dataset = create_hf_dataset(train_data)
    eval_dataset = create_hf_dataset(eval_data)
    
    # Tokenize and align labels
    logger.info("Tokenizing...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, tag2id, max_length),
        batched=True,
        remove_columns=["tokens", "tags"],
        num_proc=4,
    )
    
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, tag2id, max_length),
        batched=True,
        remove_columns=["tokens", "tags"],
        num_proc=4,
    )
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": eval_dataset,
    })
    
    return dataset_dict, tag2id, id2tag


def compute_metrics(p, id2tag: dict[int, str]):
    """Compute metrics for NER task."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        [id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = {
        "precision": seqeval.metrics.precision_score(true_labels, true_predictions),
        "recall": seqeval.metrics.recall_score(true_labels, true_predictions),
        "f1": seqeval.metrics.f1_score(true_labels, true_predictions),
    }
    
    return results


def train_model(
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    max_length: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
):
    """Main training function."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load and prepare dataset
    dataset, tag2id, id2tag = load_and_prepare_dataset(
        data_dir, tokenizer, max_length=max_length
    )
    
    # Load model with correct num_labels
    num_labels = len(tag2id)
    logger.info(f"Loading model with {num_labels} labels...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    model.to(device)
    
    # Save tag mappings
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "tag2id.json", "w") as f:
        json.dump(tag2id, f)
    with open(output_dir / "id2tag.json", "w") as f:
        json.dump({str(k): v for k, v in id2tag.items()}, f)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_steps=len(dataset["train"]) // batch_size,
        save_total_limit=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Create metrics function for this instance
    def compute_metrics_fn(p):
        return compute_metrics(p, id2tag)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Evaluate
    logger.info("Evaluating on test set...")
    eval_results = trainer.evaluate()
    logger.info(f"Eval results: {eval_results}")
    
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info(f"Training complete! Model saved to {output_dir}")
    return model, tokenizer, tag2id, id2tag


if __name__ == "__main__":
    # Check if GPU is available
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("No GPU available. Training will be slow.")

    args = parse_args()
    
    train_model(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps
    )
