#!/usr/bin/env python
"""Minimal working script to test the RAG pipeline"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import AfriQALoader
from pipeline.rag_pipeline import RAGPipeline
from evaluation.metrics import Evaluator, contains_gold
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
    golds = []
    
    for i, ex in enumerate(examples):
        print(f"\n--- Example {i+1} ---")
        print(f"Q: {ex['question']}")
        print(f"Gold: {ex['translated_answer']}")
        
        # Run RAG
        result = pipeline.run(ex['question'], return_docs=True)
        
        print(f"A: {result['answer']}")
        print(f"Confidence: {result['confidence']:.3f}" if result['confidence'] else "Confidence: N/A")
        
        # Show top retrieved doc
        if 'documents' in result and result['documents']:
            print(f"Top doc: {result['documents'][0]['text'][:100]}...")
        
        # Check if contains gold
        if contains_gold(result['answer'], ex['translated_answer']):
            print("Contains gold answer")
        else:
            print("Does NOT contain gold answer")
        
        predictions.append(result['answer'])
        golds.append(ex['translated_answer'])
    
    # 4. Evaluate
   
    
    evaluator = Evaluator()
    results = evaluator.evaluate(predictions, golds)
    
    for metric, value in results.items():
        print(f"{metric}: {value:.3f}")
    
    print("\nMinimal test completed!")


if __name__ == "__main__":
    main()