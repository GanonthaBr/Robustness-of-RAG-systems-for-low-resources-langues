#!/usr/bin/env python
"""Minimal script with enhanced evaluation"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import AfriQALoader
from pipeline.rag_pipeline import RAGPipeline
from evaluation.metrics import Evaluator


def main():
    """Run minimal test with enhanced evaluation"""
    
    # Configuration
    language = 'swa'
    num_examples = 3
    
    print("="*60)
    print("🌍 AFRI-RAG WITH ENHANCED EVALUATION")
    print("="*60)
    
    # 1. Load dataset
    print(f"\n📚 Loading AfriQA ({language})...")
    loader = AfriQALoader()
    examples = loader.load(language, split='test', num_samples=num_examples)
    
    # 2. Initialize pipeline
    pipeline = RAGPipeline(language, use_retrieval=True)
    
    # 3. Initialize evaluator
    evaluator = Evaluator(language=language)
    
    # 4. Run pipeline
    print("\n🔍 Running pipeline...")
    predictions = []
    golds_local = []
    golds_en = []
    
    for i, ex in enumerate(examples):
        print(f"\n--- Example {i+1} ---")
        print(f"Q: {ex['question']}")
        print(f"Gold (local): {ex['answers']}")
        print(f"Gold (en): {ex['translated_answer']}")
        
        # Run RAG
        result = pipeline.run(ex['question'], return_docs=True)
        
        print(f"A: {result['answer']}")
        print(f"Confidence: {result['confidence']:.3f}")
        
        # Show top doc
        if 'documents' in result and result['documents']:
            print(f"Top doc: {result['documents'][0]['text'][:100]}...")
        
        # Store for evaluation
        predictions.append(result['answer'])
        
        # Handle gold answers (AfriQA sometimes has multiple answers)
        if isinstance(ex['answers'], list):
            golds_local.append(ex['answers'][0] if ex['answers'] else "")
        else:
            golds_local.append(ex['answers'])
        
        golds_en.append(ex['translated_answer'])
    
    # 5. Evaluate with enhanced metrics
    print("\n" + "="*60)
    print("🔬 ENHANCED EVALUATION")
    print("="*60)
    
    results = evaluator.evaluate_batch(predictions, golds_local, golds_en)
    evaluator.print_summary(results)
    
    # 6. Show detailed results
    print("\n📋 DETAILED BREAKDOWN")
    print("-"*50)
    for i, detail in enumerate(results['detailed_results']):
        print(f"\nExample {i+1}:")
        print(f"  Prediction: {detail['prediction'][:100]}...")
        print(f"  Contains local gold: {detail['contains_local']}")
        print(f"  Contains English gold: {detail['contains_en']}")
        print(f"  Abstained: {detail['abstained']}")
        print(f"  Correct: {detail['correct']}")
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    main()