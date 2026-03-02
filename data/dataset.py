"""Dataset loaders for AfriQA and IrokoBench"""

from datasets import load_dataset
from typing import List, Dict, Optional


class AfriQALoader:
    """Loader for AfriQA dataset"""
    
    def __init__(self):
        self.supported_languages = ['swa', 'yor', 'kin', 'hau', 'ibo', 'bem', 'fon', 'twi', 'wol', 'zul']
    
    def load(self, language: str, split: str = 'test', num_samples: Optional[int] = None) -> List[Dict]:
        """
        Load AfriQA examples for a specific language
        
        Args:
            language: Language code (e.g., 'swa', 'yor', 'kin')
            split: 'train', 'validation', or 'test'
            num_samples: Number of samples to return (None for all)
            
        Returns:
            List of dictionaries with keys: 
            id, question, translated_question, answers, translated_answer, lang
        """
        if language not in self.supported_languages:
            raise ValueError(f"Language {language} not supported. Supported: {self.supported_languages}")
        
        dataset = load_dataset("masakhane/afriqa", language, split=split)
        
        examples = []
        for item in dataset:
            examples.append({
                'id': item['id'],
                'question': item['question'],
                'translated_question': item['translated_question'],
                'answers': item['answers'],
                'translated_answer': item['translated_answer'],
                'language': language
            })
        
        if num_samples and num_samples < len(examples):
            import random
            examples = random.sample(examples, num_samples)
        
        return examples


class IrokoBenchLoader:
    """Loader for IrokoBench AfriMMLU subset"""
    
    def __init__(self):
        self.supported_languages = ['swa', 'yor']  # Add more as needed
    
    def load(self, language: str, split: str = 'test', num_samples: Optional[int] = None) -> List[Dict]:
        """
        Load IrokoBench examples
        
        Args:
            language: Language code ('swa' or 'yor')
            split: 'train', 'validation', or 'test'
            num_samples: Number of samples to return
            
        Returns:
            List of dictionaries with question, choices, answer, language
        """
        if language not in self.supported_languages:
            raise ValueError(f"Language {language} not supported")
        
        # TODO: Implement actual IrokoBench loading once dataset is available
        # For now, return sample structure
        import random
        
        samples = []
        for i in range(min(num_samples or 10, 10)):
            samples.append({
                'id': f"{language}_{i}",
                'question': f"Sample question {i} in {language}",
                'choices': ['A', 'B', 'C', 'D'],
                'answer': random.choice(['A', 'B', 'C', 'D']),
                'language': language
            })
        
        return samples