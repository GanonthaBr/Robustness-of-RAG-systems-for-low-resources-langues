#!/usr/bin/env python
"""Minimal working script to test the RAG pipeline"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import AfriQALoader
from pipeline.rag_pipeline import RAGPipeline
from evaluation.metrics import Evaluator, contains_gold, normalize_answer_text
from dotenv import load_dotenv

# Load environment variables (HF_TOKEN, etc.) from .env

load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# Authenticate with HF Hub
from huggingface_hub import login
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token, add_to_git_credential=False)




def main():
    """Run minimal test of RAG pipeline"""
    
    # Configuration
    language = 'swa'  
    num_examples = 3 
    
   
    
    # 1. Load dataset
    print(f"\n Loading AfriQA ({language})...")
    loader = AfriQALoader()
    examples = loader.load(language, split='test', num_samples=num_examples)
    
    # 2. Initialize pipeline
    pipeline = RAGPipeline(language, use_retrieval=True)
    
    # 3. Run on examples
   
    predictions = []
    golds_local = []
    golds_translated = []
    
    for i, ex in enumerate(examples):
        # Prefer local-language gold for fair multilingual scoring.
        local_gold = normalize_answer_text(ex.get('answers', ''))
        translated_gold = normalize_answer_text(ex.get('translated_answer', ''))

        print(f"\n--- Example {i+1} ---")
        print(f"Q: {ex['question']}")
        print(f"Gold (local): {local_gold}")
        print(f"Gold (en): {translated_gold}")
        
        # Run RAG
        result = pipeline.run(ex['question'], return_docs=True)
        
        print(f"A: {result['answer']}")
        print(f"Confidence: {result['confidence']:.3f}" if result['confidence'] else "Confidence: N/A")
        
        # Show top retrieved doc
        if 'documents' in result and result['documents']:
            print(f"Top doc: {result['documents'][0]['text'][:100]}...")
        
        # Match against either local or translated gold.
        if contains_gold(result['answer'], local_gold) or contains_gold(result['answer'], translated_gold):
            print("Contains gold answer")
        else:
            print("Does NOT contain gold answer")
        
        predictions.append(result['answer'])
        golds_local.append(local_gold)
        golds_translated.append(translated_gold)
    
    # 4. Evaluate
   
    
    evaluator = Evaluator()
    results_local = evaluator.evaluate(predictions, golds_local)
    results_translated = evaluator.evaluate(predictions, golds_translated)

    print("\nMetrics vs local-language gold:")
    for metric, value in results_local.items():
        print(f"{metric}: {value:.3f}")

    print("\nMetrics vs translated (English) gold:")
    for metric, value in results_translated.items():
        print(f"{metric}: {value:.3f}")
    
    print("\nMinimal test completed!")


if __name__ == "__main__":
    main()