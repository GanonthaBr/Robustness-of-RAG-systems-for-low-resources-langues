"""Dataset loaders for AfriQA and IrokoBench"""

import pandas as pd
from datasets import Dataset
from typing import List, Dict, Optional
import random


class AfriQALoader:
    """Loader for AfriQA dataset with support for English translations"""
    
    def __init__(self):
        self.supported_languages = ['en', 'swa', 'yor', 'kin']
    
    def load(self, language: str, split: str = 'test', num_samples: Optional[int] = None) -> List[Dict]:
        """
        Load AfriQA examples for a specific language or English translation baseline.
        
        For African languages: uses native question/answers.
        For English ('en'): uses translated_question/translated_answer from Swahili AfriQA.
        This ensures fair comparison using the same dataset and questions.
        
        Args:
            language: Language code ('en', 'swa', 'yor', 'kin')
                     For 'en', loads from 'swa' but evaluates on English translations
            split: 'train', 'validation', or 'test'
            num_samples: Number of samples to return (None for all)
            
        Returns:
            List of dictionaries with keys: 
            id, question, translated_question, answers, translated_answer, language
        """
        if language not in self.supported_languages:
            raise ValueError(f"Language {language} not supported. Supported: {self.supported_languages}")
        
        # For English, load from Swahili but use translated versions
        source_language = 'swa' if language == 'en' else language
        
        # Load via HF Hub auto-converted Parquet files
        url = f"https://huggingface.co/datasets/masakhane/afriqa/resolve/refs%2Fconvert%2Fparquet/{source_language}/{split}/0000.parquet"
        dataset = Dataset.from_pandas(pd.read_parquet(url))
        
        print(f"Loaded {len(dataset)} examples for {language} ({split} split)")
        examples = []
        for idx, item in enumerate(dataset):
            if language == 'en':
                # For English baseline: use translated versions from AfriQA
                examples.append({
                    'id': idx,
                    'question': item['translated_question'],  # English translation as question
                    'translated_question': item['translated_question'],
                    'answers': item['translated_answer'],  # English translation as gold answers
                    'translated_answer': item['translated_answer'],
                    'language': 'en',
                    'translation_type': 'english_baseline_from_swahili',
                    'source_language': source_language,
                })
            else:
                # For African languages: use native language versions
                examples.append({
                    'id': idx,
                    'question': item['question'],
                    'translated_question': item['translated_question'],
                    'answers': item['answers'],
                    'translated_answer': item['translated_answer'],
                    'language': item.get('lang', language),
                    'translation_type': item.get('translation_type', ''),
                })
        
        if num_samples and num_samples < len(examples):
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