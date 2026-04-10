#!/usr/bin/env python
"""Minimal script with enhanced evaluation"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure all relative file operations (cache/results) are rooted in the repo.
os.chdir(PROJECT_ROOT)

from data.dataset import AfriQALoader
from pipeline.rag_pipeline import RAGPipeline
from evaluation.metrics import Evaluator, RetrieverEvaluator
from utils.helpers import save_json

load_dotenv(PROJECT_ROOT / '.env')


def main():
    """Run minimal test with enhanced evaluation"""
    languages = ['swa', 'yor', 'kin']
    num_examples = 100
    all_results = {}

    print("AFRI-RAG Enhanced Evaluation")
    print(f"Languages: {languages}")
    print(f"Examples per language: {num_examples}")

    for language in languages:
        print("\n")
        print(f"Language: {language}")

        # 1) Load dataset
        loader = AfriQALoader()
        examples = loader.load(language, split='test', num_samples=num_examples)

        # 2) Initialize pipeline
        pipeline = RAGPipeline(language, use_retrieval=True)

        # 3) Initialize evaluator
        evaluator = Evaluator(language=language)

        # 4) Run pipeline over this language's examples
        predictions = []
        golds_local = []
        golds_en = []
        all_retrieved_docs = []

        for i, ex in enumerate(examples):
            print(f"\nExample {i + 1}/{len(examples)}")
            print(f"Q: {ex['question']}")

            result = pipeline.run(ex['question'], return_docs=True)
            predictions.append(result['answer'])
            
            # Capture retrieved documents for retriever evaluation
            if 'docs' in result and result['docs']:
                all_retrieved_docs.append(result['docs'])
            else:
                all_retrieved_docs.append([])

            answers = ex.get('answers', "")
            if isinstance(answers, list):
                golds_local.append(answers[0] if answers else "")
            else:
                golds_local.append(answers)

            golds_en.append(ex.get('translated_answer', ""))

        # 5) Evaluate retriever quality (before LLM)
        retriever_metrics = RetrieverEvaluator.evaluate_retrieval(all_retrieved_docs)
        print(f"\n{'='*70}")
        print(f"Retriever Metrics for {language}:")
        print(f"  Mean Similarity (all docs): {retriever_metrics['mean_similarity']:.4f}")
        print(f"  Mean Top-1 Similarity: {retriever_metrics['mean_top1_similarity']:.4f}")
        print(f"  Mean Top-5 Similarity: {retriever_metrics['mean_top5_similarity']:.4f}")
        print(f"  Max Similarity: {retriever_metrics['max_similarity']:.4f}")
        print(f"  Min Similarity: {retriever_metrics['min_similarity']:.4f}")
        print(f"{'='*70}")

        # 6) Evaluate generation quality (predictions vs gold)
        results = evaluator.evaluate_batch(predictions, golds_local, golds_en)
        evaluator.print_summary(results)

        # Keep a compact copy for final save/print (including retriever metrics)
        all_results[language] = {
            'retriever_metrics': retriever_metrics,
            'generation_metrics': {
                'contains_gold_local': results['contains_gold_local'],
                'contains_gold_english': results['contains_gold_english'],
                'abstention_rate': results['abstention_rate'],
                'correct_rate': results['correct_rate'],
                'precision_on_answered': results['precision_on_answered'],
                'num_samples': results['num_samples'],
                'num_abstained': results['num_abstained'],
            }
        }

        save_json(results, str(PROJECT_ROOT / f"results/{language}_enhanced_results.json"))

    print("\n")
    print("="*70)
    print("FINAL SUMMARY BY LANGUAGE")
    print("="*70)
    for language, lang_results in all_results.items():
        gen_metrics = lang_results.get('generation_metrics', {})
        ret_metrics = lang_results.get('retriever_metrics', {})
        
        print(f"\n{language.upper()}:")
        print(f"  Retriever: sim={ret_metrics.get('mean_similarity', 0):.4f}, top1={ret_metrics.get('mean_top1_similarity', 0):.4f}, top5={ret_metrics.get('mean_top5_similarity', 0):.4f}")
        print(f"  Generation: correct={gen_metrics.get('correct_rate', 0):.1%}, local={gen_metrics.get('contains_gold_local', 0):.1%}, en={gen_metrics.get('contains_gold_english', 0):.1%}, abstain={gen_metrics.get('abstention_rate', 0):.1%}")

    save_json(all_results, str(PROJECT_ROOT / "results/all_languages_enhanced_summary.json"))


if __name__ == "__main__":
    main()