# XLM-RoBERTa NER Training

Training script for Named Entity Recognition (NER) using XLM-RoBERTa-large on multilingual CONLL-formatted data.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Data Format

The training script expects CONLL format files in the `../annotation_samples_klara/` directory:

```
TOKEN TAG
John B-PER
Smith I-PER
lives O
in O
New B-LOC
York I-LOC
. O

Another O
sentence O
```

Each line contains a token and its BIO tag, separated by whitespace. Sentences are separated by blank lines.

## Training

### Basic Training

```bash
python train_ner.py
```

### Custom Configuration

Edit these variables in `train_ner.py`:

- `MODEL_NAME`: Model to fine-tune (default: `xlm-roberta-large`)
- `MAX_LENGTH`: Maximum sequence length (default: 512)
- `BATCH_SIZE`: Batch size (default: 8)
- `EPOCHS`: Number of training epochs (default: 3)
- `LEARNING_RATE`: Learning rate (default: 2e-5)
- `OUTPUT_DIR`: Where to save the model (default: `xlm_roberta_ner_model/`)

### Training on GPU

The script automatically uses GPU if available. Check:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Inference

### Using the Trained Model

```python
from inference_ner import NERInference
from pathlib import Path

model_dir = Path("xlm_roberta_ner_model")
ner = NERInference(model_dir)

# Single prediction
result = ner.predict("John Smith lives in New York.")
print(result['entities'])

# Batch predictions
texts = ["Text 1", "Text 2"]
results = ner.predict_batch(texts)
```

### Command Line

```bash
python inference_ner.py
```

## Model Outputs

After training, the `xlm_roberta_ner_model/` directory contains:

- `pytorch_model.bin`: Fine-tuned model weights
- `config.json`: Model configuration
- `tokenizer.json`: Tokenizer configuration
- `tag2id.json`: Tag to ID mappings
- `id2tag.json`: ID to tag mappings
- `eval_results.json`: Final evaluation metrics
- `checkpoint-*/`: Saved checkpoints during training

## Metrics

The training script reports:

- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)
- **F1 Score**: Harmonic mean of precision and recall

These are computed at the entity level using `seqeval`.

## Supported Languages

XLM-RoBERTa supports 100+ languages. The model can handle:

- Czech (cs)
- Dutch (nl)
- English (en)
- German (de)
- Danish (da)
- Hungarian (hu)
- Polish (pl)
- Slovak (sk)
- Serbian (sr)
- Croatian (hr)

## Performance Tips

1. **Increase batch size** if you have enough GPU memory (8 → 16 or 32)
2. **More data**: Train on more examples for better performance
3. **Longer training**: Increase epochs if validation metrics improve
4. **Learning rate**: 2e-5 works well for most cases, try 1e-5 to 5e-5
5. **Warmup steps**: Increase if training is unstable

## Troubleshooting

### Out of Memory (OOM)

Reduce `BATCH_SIZE` or `MAX_LENGTH`:

```python
BATCH_SIZE = 4
MAX_LENGTH = 256
```

### Training is slow

- Check if GPU is being used: `nvidia-smi`
- Increase `BATCH_SIZE` if memory allows
- Reduce `MAX_LENGTH`

### Poor performance

- Ensure CONLL format is correct
- Check for class imbalance in tags
- Increase training data
- Increase epochs
- Try lower learning rate

## References

- [XLM-RoBERTa Model Card](https://huggingface.co/xlm-roberta-large)
- [Hugging Face Token Classification](https://huggingface.co/docs/transformers/tasks/token_classification)
- [CONLL Format](https://www.clips.uantwerpen.be/conll2003/ner/)
- [seqeval Documentation](https://github.com/chakki-works/seqeval)
