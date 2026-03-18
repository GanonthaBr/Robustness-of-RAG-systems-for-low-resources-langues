"""Evaluation metrics for RAG output"""

from typing import List, Dict
import ast
import numpy as np
from sklearn.metrics import f1_score


def normalize_answer_text(value) -> str:
    """Normalize answer values into comparable plain text."""
    if isinstance(value, str):
        text = value.strip()
        # Handle stringified lists such as "['1992']" from dataset fields.
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)) and parsed:
                    return str(parsed[0]).strip()
            except (ValueError, SyntaxError):
                pass
        return text

    if isinstance(value, (list, tuple)):
        return str(value[0]).strip() if value else ""

    return str(value).strip()


def exact_match(prediction: str, gold: str, normalize: bool = True) -> bool:
    """Exact match between prediction and gold answer"""
    prediction = normalize_answer_text(prediction)
    gold = normalize_answer_text(gold)
    if normalize:
        prediction = prediction.lower().strip()
        gold = gold.lower().strip()
    return prediction == gold


def f1_score_answer(prediction: str, gold: str) -> float:
    """Token-level F1 score between prediction and gold"""
    prediction = normalize_answer_text(prediction)
    gold = normalize_answer_text(gold)
    pred_tokens = set(prediction.lower().split())
    gold_tokens = set(gold.lower().split())
    
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    
    common = pred_tokens.intersection(gold_tokens)
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def contains_gold(prediction: str, gold: str) -> bool:
    """Check if gold answer appears in prediction"""
    prediction = normalize_answer_text(prediction)
    gold = normalize_answer_text(gold)
    return gold.lower() in prediction.lower()


class Evaluator:
    """Evaluate RAG outputs against gold answers"""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate(self, predictions: List[str], golds: List[str]) -> Dict:
        """
        Evaluate predictions against gold answers
        
        Returns:
            Dict with EM, F1, and contains scores
        """
        em_scores = []
        f1_scores = []
        contains_scores = []
        
        for pred, gold in zip(predictions, golds):
            em_scores.append(float(exact_match(pred, gold)))
            f1_scores.append(f1_score_answer(pred, gold))
            contains_scores.append(float(contains_gold(pred, gold)))
        
        return {
            'exact_match': np.mean(em_scores),
            'f1': np.mean(f1_scores),
            'contains_gold': np.mean(contains_scores),
            'num_samples': len(predictions)
        }