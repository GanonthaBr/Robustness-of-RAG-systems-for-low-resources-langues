"""Enhanced evaluation metrics for African-language RAG"""

import re
import numpy as np
from typing import List, Dict, Union, Optional


class Evaluator:
    """Improved evaluator for African-language RAG with language-aware matching"""
    
    def __init__(self, language: Optional[str] = None):
        """
        Args:
            language: Language code for language-specific abstention phrases
        """
        self.language = language
        
        # Language-specific abstention phrases
        self.abstention_phrases = {
            'swa': ['sijui', 'siijui', 'sijui', 'sijui jibu', 'sijui la kujibu'],
            'yor': ['mi o', 'emi o', 'mi o mọ', 'emi o mọ', 'nko mọ'],
            'kin': ['ntabwo nzi', 'simenzi', 'ntabwo menzi', 'sinzi', 'ntabwo mbizi'],
            'en': ['i don\'t know', 'unknown', 'not sure', 'i do not know', 'cannot answer']
        }
        
        # Default to empty list if language not found
        self.abstention_list = self.abstention_phrases.get(language, [])
    
    def extract_numbers(self, text: str) -> List[str]:
        """Extract all numbers from text"""
        return re.findall(r'\d+', text)
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Convert to lowercase and strip
        text = text.lower().strip()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove punctuation (keep numbers and letters)
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def contains_entity(self, prediction: str, gold: str) -> bool:
        """
        Check if gold entity appears in prediction (flexible matching)
        This is your primary metric for African-language RAG
        """
        # Normalize both texts
        pred_norm = self.normalize_text(prediction)
        gold_norm = self.normalize_text(gold)
        
        # Direct containment
        if gold_norm in pred_norm:
            return True
        
        # For multi-word gold, check if key words appear
        gold_words = gold_norm.split()
        if len(gold_words) > 1:
            # Count how many gold words appear in prediction
            matches = sum(1 for word in gold_words if word in pred_norm)
            # If more than half the words match, consider it correct
            if matches >= len(gold_words) / 2:
                return True
        
        # For numeric answers
        if gold_norm.isdigit():
            pred_numbers = self.extract_numbers(pred_norm)
            return gold_norm in pred_numbers
        
        return False
    
    def check_abstention(self, text: str) -> bool:
        """Check if the model abstained from answering"""
        text_lower = text.lower()
        
        # Check language-specific abstention phrases
        for phrase in self.abstention_list:
            if phrase in text_lower:
                return True
        
        # Also check for very short responses (likely abstention)
        if len(text.strip()) < 5:
            return True
        
        return False
    
    def evaluate_single(self, prediction: str, gold_local: str, 
                       gold_en: Optional[str] = None) -> Dict:
        """
        Evaluate a single prediction against gold answers
        
        Args:
            prediction: Model-generated answer
            gold_local: Gold answer in the query language
            gold_en: Gold answer in English (optional)
        
        Returns:
            Dictionary with evaluation results
        """
        # Check abstention first
        abstained = self.check_abstention(prediction)
        
        # If abstained, no need to check correctness
        if abstained:
            return {
                'contains_local': False,
                'contains_en': False if gold_en else None,
                'abstained': True,
                'correct': False,
                'prediction': prediction,
                'gold_local': gold_local,
                'gold_en': gold_en
            }
        
        # Check if contains gold
        contains_local = self.contains_entity(prediction, gold_local)
        
        contains_en = None
        if gold_en:
            contains_en = self.contains_entity(prediction, gold_en)
        
        # For questions where gold is identical in both languages
        # (e.g., numbers, proper nouns), we can use either
        correct = contains_local or (contains_en if contains_en is not None else False)
        
        return {
            'contains_local': contains_local,
            'contains_en': contains_en,
            'abstained': abstained,
            'correct': correct,
            'prediction': prediction,
            'gold_local': gold_local,
            'gold_en': gold_en
        }
    
    def evaluate_batch(self, predictions: List[str], golds_local: List[str],
                      golds_en: Optional[List[str]] = None) -> Dict:
        """
        Evaluate multiple predictions
        
        Args:
            predictions: List of model-generated answers
            golds_local: List of gold answers in query language
            golds_en: Optional list of gold answers in English
        
        Returns:
            Dictionary with aggregate metrics
        """
        if golds_en and len(golds_en) != len(predictions):
            raise ValueError("golds_en must have same length as predictions")
        
        results = []
        for i, pred in enumerate(predictions):
            gold_en = golds_en[i] if golds_en else None
            results.append(self.evaluate_single(pred, golds_local[i], gold_en))
        
        # Calculate aggregate metrics
        contains_local_rate = np.mean([r['contains_local'] for r in results])
        contains_en_rate = np.mean([r['contains_en'] for r in results if r['contains_en'] is not None])
        abstention_rate = np.mean([r['abstained'] for r in results])
        correct_rate = np.mean([r['correct'] for r in results])
        
        # Calculate precision on non-abstained examples
        non_abstained = [r for r in results if not r['abstained']]
        if non_abstained:
            precision = np.mean([r['correct'] for r in non_abstained])
        else:
            precision = 0.0
        
        return {
            'contains_gold_local': contains_local_rate,
            'contains_gold_english': contains_en_rate,
            'abstention_rate': abstention_rate,
            'correct_rate': correct_rate,
            'precision_on_answered': precision,
            'num_samples': len(results),
            'num_abstained': sum(r['abstained'] for r in results),
            'detailed_results': results
        }
    
    def print_summary(self, results: Dict):
        """Print a nice summary of evaluation results"""
       
        print("EVALUATION SUMMARY")
      
        print(f"Samples: {results['num_samples']}")
        print(f"Abstained: {results['num_abstained']} ({results['abstention_rate']:.1%})")
        print(f"\n Correct rate (overall): {results['correct_rate']:.1%}")
        print(f"Precision (when answered): {results['precision_on_answered']:.1%}")
        print(f"\n Contains gold (local): {results['contains_gold_local']:.1%}")
        if results['contains_gold_english'] is not None:
            print(f" Contains gold (English): {results['contains_gold_english']:.1%}")
        

# For backward compatibility, keep original functions if needed
def exact_match(prediction: str, gold: str, normalize: bool = True) -> bool:
    """Original exact match function (kept for compatibility)"""
    if normalize:
        prediction = prediction.lower().strip()
        gold = gold.lower().strip()
    return prediction == gold


def f1_score_answer(prediction: str, gold: str) -> float:
    """Original F1 function (kept for compatibility)"""
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
    """Original contains function (kept for compatibility)"""
    return gold.lower() in prediction.lower()