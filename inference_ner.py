"""
Inference script for trained XLM-RoBERTa NER model.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


class NERInference:
    def __init__(self, model_dir: Path):
        """Load trained model and tokenizer."""
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tag mappings
        with open(self.model_dir / "id2tag.json") as f:
            id2tag = json.load(f)
            self.id2tag = {int(k): v for k, v in id2tag.items()}
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()
        
        # Create pipeline
        self.nlp = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            aggregation_strategy="simple",  # or "first", "average"
        )
    
    def predict(self, text: str) -> Dict:
        """Predict NER tags for input text."""
        results = self.nlp(text)
        
        # Group by entity
        entities = []
        for result in results:
            if result["entity"] != "O":
                entities.append({
                    "text": result["word"],
                    "entity": result["entity"],
                    "score": result["score"],
                    "start": result["start"],
                    "end": result["end"],
                })
        
        return {
            "text": text,
            "entities": entities,
            "raw_results": results,
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict NER tags for multiple texts."""
        return [self.predict(text) for text in texts]


if __name__ == "__main__":
    # Example usage
    model_dir = Path(__file__).parent / "xlm_roberta_ner_model"
    
    ner = NERInference(model_dir)
    
    # Single prediction
    text = "John Smith lives in New York and works at Google."
    result = ner.predict(text)
    print(f"Text: {result['text']}")
    print(f"Entities: {result['entities']}")
    
    # Batch prediction
    texts = [
        "Albert Einstein was a physicist.",
        "The meeting is on March 15, 2024 in Berlin.",
    ]
    batch_results = ner.predict_batch(texts)
    for result in batch_results:
        print(f"\nText: {result['text']}")
        print(f"Entities: {result['entities']}")
