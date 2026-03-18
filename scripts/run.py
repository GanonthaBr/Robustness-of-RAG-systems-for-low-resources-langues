#!/usr/bin/env python
"""Minimal script with enhanced evaluation"""

import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import AfriQALoader
from pipeline.rag_pipeline import RAGPipeline
from evaluation.metrics import Evaluator
from utils.helpers import save_json


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))


def main():
    """Run minimal test with enhanced evaluation"""
    languages = ['swa', 'yor', 'kin']
    num_examples = 50
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

        for i, ex in enumerate(examples):
            print(f"\nExample {i + 1}/{len(examples)}")
            print(f"Q: {ex['question']}")

            result = pipeline.run(ex['question'], return_docs=True)
            predictions.append(result['answer'])

            answers = ex.get('answers', "")
            if isinstance(answers, list):
                golds_local.append(answers[0] if answers else "")
            else:
                golds_local.append(answers)

            golds_en.append(ex.get('translated_answer', ""))

        # 5) Evaluate this language and print summary
        results = evaluator.evaluate_batch(predictions, golds_local, golds_en)
        evaluator.print_summary(results)

        # Keep a compact copy for final save/print
        all_results[language] = {
            'contains_gold_local': results['contains_gold_local'],
            'contains_gold_english': results['contains_gold_english'],
            'abstention_rate': results['abstention_rate'],
            'correct_rate': results['correct_rate'],
            'precision_on_answered': results['precision_on_answered'],
            'num_samples': results['num_samples'],
            'num_abstained': results['num_abstained'],
        }

        save_json(results, os.path.join(PROJECT_ROOT, f"results/{language}_enhanced_results.json"))

    print("\n")
    print("Final summary by language")
    for language, metrics in all_results.items():
        print(
            f"{language}: correct={metrics['correct_rate']:.1%}, "
            f"local={metrics['contains_gold_local']:.1%}, "
            f"en={metrics['contains_gold_english']:.1%}, "
            f"abstain={metrics['abstention_rate']:.1%}"
        )

    save_json(all_results, os.path.join(PROJECT_ROOT, "results/all_languages_enhanced_summary.json"))


if __name__ == "__main__":
    main()