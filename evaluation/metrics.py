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


class RetrieverEvaluator:
    """Evaluate retriever quality based on similarity scores"""
    
    @staticmethod
    def evaluate_retrieval(retrieved_docs: List[List[Dict]]) -> Dict:
        """
        Evaluate retriever using similarity scores of retrieved documents
        
        Args:
            retrieved_docs: List of lists, where each inner list contains dicts with 'score' key
                          retrieved_docs[i] = list of top-k docs for query i, each with similarity score
        
        Returns:
            Dict with retriever metrics:
            - mean_similarity: average similarity across all retrieved docs
            - mean_top1_similarity: average of best match per query
            - mean_top5_similarity: average of best 5 matches per query
            - max_similarity: best similarity found
            - min_similarity: worst similarity found
        """
        if not retrieved_docs or all(len(docs) == 0 for docs in retrieved_docs):
            return {
                'mean_similarity': 0.0,
                'mean_top1_similarity': 0.0,
                'mean_top5_similarity': 0.0,
                'max_similarity': 0.0,
                'min_similarity': 0.0,
                'num_queries': len(retrieved_docs)
            }
        
        all_similarities = []
        top1_similarities = []
        top5_similarities = []
        
        for docs in retrieved_docs:
            if docs:
                scores = [doc['score'] for doc in docs]
                all_similarities.extend(scores)
                top1_similarities.append(scores[0])
                top5_similarities.append(np.mean(scores[:min(5, len(scores))]))
        
        if not all_similarities:
            return {
                'mean_similarity': 0.0,
                'mean_top1_similarity': 0.0,
                'mean_top5_similarity': 0.0,
                'max_similarity': 0.0,
                'min_similarity': 0.0,
                'num_queries': len(retrieved_docs)
            }
        
        return {
            'mean_similarity': float(np.mean(all_similarities)),
            'mean_top1_similarity': float(np.mean(top1_similarities)) if top1_similarities else 0.0,
            'mean_top5_similarity': float(np.mean(top5_similarities)) if top5_similarities else 0.0,
            'max_similarity': float(np.max(all_similarities)),
            'min_similarity': float(np.min(all_similarities)),
            'num_queries': len(retrieved_docs)
        }


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