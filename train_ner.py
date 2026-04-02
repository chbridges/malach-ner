"""
Train XLM-RoBERTa-large on NER dataset using the transformers library.
Supports CoNLL format input.
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
from datasets import Dataset, DatasetDict, concatenate_datasets
import seqeval.metrics

IGNORE_TAGS = ("AMBIG", "TERM")
TAGSETS = {
    "ehri": ['B-CAMP', 'B-DATE', 'B-GHETTO', 'B-LOC', 'B-ORG', 'B-PERS', 'I-CAMP', 'I-DATE', 'I-GHETTO', 'I-LOC', 'I-ORG', 'I-PERS', 'O'],
    "malach": ['B-CAMP', 'B-DATE', 'B-GHETTO', 'B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-CAMP', 'I-DATE', 'I-GHETTO', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O'],
    "date_extraction": ['B-DATE', 'I-DATE', 'O'],
}

EHRI = Path("./ehri/")
MALACH = Path("./final/training_data")
BOTH = [EHRI, MALACH]

MODEL_MAP = {
    # MalachNER paper
    "large": "FacebookAI/xlm-roberta-large",
    "ehri": "ehri-ner/xlm-roberta-large-ehri-ner-all",
    # XLM-RoBERTa-malach paper
    "malach1": "ufal/xlm-roberta-malach",  # formerly ChrisBridges/xlm-r-malach-v5-1e-5
    "malach2": "ChrisBridges/xlm-r-malach-v5-2e-5",
    "holobert": "Isuri97/holo_mlm_bert",
    # Miscellaneous experiments
    "date_extraction": "ufal/xlm-roberta-malach",
    }

EXPERIMENTS = {
    "large": {
        "ehri_ehri": {"train_dir": EHRI, "test_dir": EHRI, "tagset": "ehri"},
        "ehri_malach": {"train_dir": EHRI, "test_dir": MALACH, "tagset": "ehri"},
        "malach_ehri": {"train_dir": MALACH, "test_dir": EHRI, "tagset": "ehri"},
        "malach_malach": {"train_dir": MALACH, "test_dir": MALACH, "tagset": "malach"},
        "both_ehri": {"train_dir": BOTH, "test_dir": EHRI, "tagset": "ehri"},
        "both_malach": {"train_dir": BOTH, "test_dir": MALACH, "tagset": "malach"},
    },
    "ehri": {
        "frozen_ehri": {"train_dir": None, "test_dir": EHRI, "tagset": "ehri"},
        "frozen_malach": {"train_dir": None, "test_dir": MALACH, "tagset": "ehri"},
        "finetune_ehri": {"train_dir": MALACH, "test_dir": EHRI, "tagset": "ehri"},
        "finetune_malach": {"train_dir": MALACH, "test_dir": MALACH, "tagset": "ehri"},
    },
    "malach": {
        "finetune_ehri": {"train_dir": EHRI, "test_dir": EHRI, "tagset": "ehri"},
    },
    "holobert": {
        "finetune_ehri": {"train_dir": EHRI, "test_dir": EHRI, "tagset": "ehri"},
    },
    "date_extraction": {
        "date_extraction": {"train_dir": BOTH, "test_dir": EHRI, "tagset": "date_extraction"},
    },
}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
def parse_args():
    parser = argparse.ArgumentParser(description="Train XLM-RoBERTa for NER")
    parser.add_argument("--experiment", type=str, choices=EXPERIMENTS.keys(), help="Experiment name")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save model and outputs")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data to use for training")
    parser.add_argument("--eval_model", type=Path, help="Local fine-tuned model path for per-language evaluation (overrides all previous arguments)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    return parser.parse_args()


def read_conll_file(file_path: str, tagset: list) -> list[tuple[list[str], list[str]]]:
    """Read CoNLL format file and return list of (tokens, tags) tuples."""
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

                    # map between EHRI-NER and MalachNER
                    if tag.endswith("PER") and "B-PERS" in tagset:
                        tag = tag.replace("PER", "PERS")
                    elif tag.endswith("PERS") and "B-PER" in tagset:
                        tag = tag.replace("PERS", "PER")
                    elif tag not in tagset:
                        tag = "O"

                    tokens.append(token)
                    tags.append(tag)
    
    if tokens:
        sentences.append((tokens, tags))
    
    return sentences


def tokenize_and_align_labels(
    examples: dict,
    tokenizer,
    label2id: dict[str, int],
) -> dict:
    """Tokenize text and align labels with subword tokens."""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
    )
    
    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def load_and_prepare_dataset(
    data_dir: Path,
    tokenizer,
    splits: list[str],
    tagset: list[str],
    language: str = "",
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[DatasetDict, dict[str, int], dict[int, str]]:
    """Load CoNLL files and prepare HuggingFace dataset."""
    
    logger.info("Loading CoNLL files...")

    # Load files by split, optionally filter by language
    files = {split: [] for split in splits}
    for split in splits:
        files[split] = [f for f in data_dir.glob(f"{language}*.conll") if split in f.name]

    data = {split: [] for split in splits}
    
    for split in splits:
        for conll_file in files[split]:
            logger.info(f"Loading {conll_file.name}...")
            content = read_conll_file(str(conll_file), tagset)
            data[split].extend(content)
            logger.info(f"  Loaded {len(data[split])} sentences")

    all_training_data = sum(data.values(), [])
        
    # Extract unique tags
    tag_list = tagset
    label2id = {label: i for i, label in enumerate(tagset)}
    id2label = {i: label for i, label in enumerate(tagset)}
    
    logger.info(f"Found {len(tag_list)} unique tags: {tag_list}")
    logger.info(f"Label mappings: {label2id}")
    
    # Split into train/eval
    if "train" in data and len(data["dev"]) == 0:
        np.random.seed(seed)
        indices = np.random.permutation(len(all_training_data))
        split_idx = int(len(all_training_data) * train_ratio)
        
        data["train"] = [all_training_data[i] for i in indices[:split_idx]]
        data["dev"] = [all_training_data[i] for i in indices[split_idx:]]
        
        logger.info(f"Train: {len(data['train'])}, Dev: {len(data['dev'])}")
        
    # Convert to HF datasets format
    def create_hf_dataset(data: list[tuple[list[str], list[str]]]):
        tokens_list = [item[0] for item in data]
        tags_list = [item[1] for item in data]
        return Dataset.from_dict({
            "tokens": tokens_list,
            "tags": tags_list,
        })

    tokenized = {split: create_hf_dataset(data[split]).map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
        batched=True,
        remove_columns=["tokens", "tags"],
        num_proc=4,
    ) for split in splits}
    
    dataset_dict = DatasetDict({
        split: tokenized[split] for split in splits
    })
    
    return dataset_dict, label2id, id2label


def compute_metrics(p, label_list: list[str]):
    """Compute metrics for NER task."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = {
        "precision": seqeval.metrics.precision_score(true_labels, true_predictions),
        "recall": seqeval.metrics.recall_score(true_labels, true_predictions),
        "f1": seqeval.metrics.f1_score(true_labels, true_predictions),
        "report": seqeval.metrics.classification_report(true_labels, true_predictions),
    }
    
    return results


def train_model(
    model_name: str,
    exp_config: dict,
    output_dir: Path,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    handle: str,
):
    """Main training function."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    np.random.seed(seed)

    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load and prepare datasets
    train_dirs = exp_config["train_dir"]
    test_dir = exp_config["test_dir"]
    label_list = TAGSETS[exp_config["tagset"]]

    test_dataset, label2id, id2label = load_and_prepare_dataset(
        test_dir, tokenizer, ["test"], label_list
    )

    train_dirs = train_dirs if isinstance(train_dirs, list) else [train_dirs]

    if train_dirs[0] is not None:
        train_parts = []
        dev_parts = []
        for train_dir in train_dirs:
            training_parts, _, _ = load_and_prepare_dataset(
                train_dir, tokenizer, ["train", "dev"], label_list
            )
            train_parts.append(training_parts["train"])
            dev_parts.append(training_parts["dev"])

        training_dataset = DatasetDict({
            "train": concatenate_datasets(train_parts),
            "dev": concatenate_datasets(dev_parts),
        })
    else:
        training_dataset = None
    
    # Load model with correct num_labels
    num_labels = len(label_list)
    logger.info(f"Loading model with {num_labels} labels...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)
    
    # Save label mappings
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "label2id.json", "w") as f:
        json.dump(label2id, f)
    with open(output_dir / "id2label.json", "w") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch" if training_dataset else "no",
        save_strategy="epoch" if training_dataset else "no",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    def compute_metrics_fn(p):
        return compute_metrics(p, label_list)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset["train"] if training_dataset else None,
        eval_dataset=training_dataset["dev"] if training_dataset else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )
    
    # Train
    if training_dataset is None:
        logger.info("Skipping training (already trained)")
    else:
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        logger.info(f"Saving model to {output_dir}...")
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
    
    # Evaluate
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset["test"], metric_key_prefix="test")
    test_report = test_results.pop("test_report")
    logger.info(f"Eval results: {test_results}")
    logger.info(f"\n{test_report}")
    
    with open(output_dir / f"{handle}.json", "w") as f:
        json.dump(test_results, f, indent=2)
    with open(output_dir / f"{handle}.txt", "w") as f:
        f.write(test_report)
    
    return model, tokenizer, label2id, id2label


def evaluate_per_language(local_model_path: Path, batch_size: int, data_path: Path):
    # set device and defaults
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    label_list = TAGSETS["malach"]
    num_labels = len(label_list)
    languages = sorted(set(str(p)[:2] for p in data_path.iterdir()))

    def compute_metrics_fn(p):
        return compute_metrics(p, label_list)

    # init local model
    logger.info(f"Loading local tokenizer...")
    model = None
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = None
    
    for language in languages:
        logger.info(f"Evaluating {language}")

        dataset, label2id, id2label = load_and_prepare_dataset(
            data_path, tokenizer, ["test"], label_list, language=language
        )
        test_dataset = dataset["test"]

        if model is None:
            logger.info(f"Loading model with {num_labels} labels...")

            model = AutoModelForTokenClassification.from_pretrained(
                local_model_path,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
            )
            model.to(device)

        # Evaluate
        if trainer is None:
            logger.info(f"Initializing Trainer...")
            trainer = Trainer(
                model=model,
                args=TrainingArguments(per_device_eval_batch_size=batch_size),
                eval_dataset=test_dataset,
                processing_class=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics_fn,
            )
        else:
            trainer.eval_dataset = test_dataset

        logger.info(f"Evaluating on {language} test set...")
        test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")
        test_report = test_results.pop("test_report")
        logger.info(f"Eval results: {test_results}")
        logger.info(f"\n{test_report}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("No GPU available. Training will be slow.")

    args = parse_args()

    if args.eval_model:
        logger.info("Per-language evaluating EHRI-NER")
        evaluate_per_language(args.eval_model, args.batch_size, EHRI / "language_test_splits")

        logger.info("Per-language evaluating MalachNER")
        evaluate_per_language(args.eval_model, args.batch_size, MALACH)

        exit(0)

    model_type = "malach" if "malach" in args.experiment else args.experiment  # covers malach1 and malach2
    
    for exp_name, exp_config in EXPERIMENTS[model_type].items():
        for seed in (0, 42, 1234):
            if exp_config["train_dir"] is None and seed != 0:
                continue
            if exp_name == "date_extraction" and seed != 42:
                pass

            output_dir = Path(args.output_dir) / args.experiment
            handle=f"{exp_name}_{seed}"
            if (output_dir / f"{handle}.txt").exists():
                logger.info(f"Skipping experiment {args.experiment}/{handle} (already exists)")
                continue

            logger.info(f"Running experiment: {exp_name}, seed: {seed}")
            train_model(
                model_name=MODEL_MAP[args.experiment],
                exp_config=exp_config,
                output_dir=output_dir,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                seed=seed,
                handle=handle,
        )
