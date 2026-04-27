#!/usr/bin/env python
"""Compare retriever performance across embedding models"""

import sys
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

from data.dataset import AfriQALoader
from retrieval.dense_retriever import DenseRetriever
from pipeline.rag_pipeline import RAGPipeline
from evaluation.metrics import Evaluator, RetrieverEvaluator
from config.settings import EMBEDDING_MODELS
from utils.helpers import save_json

load_dotenv(PROJECT_ROOT / '.env')


def main():
    """Run evaluation with different embedding models"""
    languages = ['swa', 'yor', 'kin']
    num_examples = 50
    embedding_models = EMBEDDING_MODELS  # {'e5-base': ..., 'qwen3': ...}
    comparison_results = {}

    print("="*70)
    print("EMBEDDING MODEL COMPARISON")
    print("="*70)
    print(f"Languages: {languages}")
    print(f"Examples per language: {num_examples}")
    print(f"Models: {list(embedding_models.keys())}")
    print("="*70)

    for model_name, model_path in embedding_models.items():
        print(f"\n{'#'*70}")
        print(f"# Testing: {model_name} ({model_path})")
        print(f"{'#'*70}")
        
        model_results = {}
        
        for language in languages:
            print(f"\n{'='*70}")
            print(f"Language: {language} | Model: {model_name}")
            print(f"{'='*70}")

            # 1) Load dataset
            loader = AfriQALoader()
            examples = loader.load(language, split='test', num_samples=num_examples)

            # 2) Initialize retriever with specific embedding model
            print(f"Initializing retriever with {model_name}...")
            retriever = DenseRetriever(model_name=model_path)

            # 3) Load and index Wikipedia corpus
            from data.wikipedia import WikipediaCorpus
            corpus = WikipediaCorpus(language=language, cache_dir=str(PROJECT_ROOT / "cache"))
            passages = corpus.get_passages()
            retriever.index_corpus(passages)

            # 4) Initialize evaluator
            evaluator = Evaluator(language=language)

            # 5) Run retrieval on examples and collect metrics
            all_retrieved_docs = []
            predictions = []
            golds_local = []
            golds_en = []

            for i, ex in enumerate(examples):
                if (i + 1) % 10 == 0:
                    print(f"  Processing: {i + 1}/{len(examples)}")

                question = ex['question']
                
                # Retrieve documents
                retrieved = retriever.retrieve(question, k=10)
                all_retrieved_docs.append(retrieved)

                # For now, just use top doc as "prediction" (focus on retrieval)
                if retrieved:
                    predictions.append(retrieved[0].get('text', '')[:200])
                else:
                    predictions.append('')

                answers = ex.get('answers', "")
                if isinstance(answers, list):
                    golds_local.append(answers[0] if answers else "")
                else:
                    golds_local.append(answers)

                golds_en.append(ex.get('translated_answer', ""))

            # 6) Evaluate retriever quality
            retriever_metrics = RetrieverEvaluator.evaluate_retrieval(all_retrieved_docs)
            
            print(f"\n{'='*70}")
            print(f"Retriever Metrics ({model_name} - {language}):")
            print(f"{'='*70}")
            print(f"  Mean Similarity (all docs): {retriever_metrics['mean_similarity']:.4f}")
            print(f"  Mean Top-1 Similarity: {retriever_metrics['mean_top1_similarity']:.4f}")
            print(f"  Mean Top-5 Similarity: {retriever_metrics['mean_top5_similarity']:.4f}")
            print(f"  Max Similarity: {retriever_metrics['max_similarity']:.4f}")
            print(f"  Min Similarity: {retriever_metrics['min_similarity']:.4f}")
            print(f"{'='*70}")

            # Store results
            model_results[language] = {
                'retriever_metrics': retriever_metrics,
                'num_examples': len(examples)
            }

            # Save per-model, per-language results
            save_json(
                model_results[language],
                str(PROJECT_ROOT / f"results/{model_name}_{language}_retriever_metrics.json")
            )

        comparison_results[model_name] = model_results

    # Final comparison summary
    print(f"\n{'#'*70}")
    print(f"# FINAL COMPARISON SUMMARY")
    print(f"{'#'*70}")
    
    for language in languages:
        print(f"\n{language.upper()}:")
        print(f"{'  Model':<15} {'Mean Sim':<12} {'Top-1':<12} {'Top-5':<12}")
        print(f"  {'-'*50}")
        
        for model_name, model_results in comparison_results.items():
            if language in model_results:
                ret_m = model_results[language]['retriever_metrics']
                print(
                    f"  {model_name:<15} "
                    f"{ret_m['mean_similarity']:<12.4f} "
                    f"{ret_m['mean_top1_similarity']:<12.4f} "
                    f"{ret_m['mean_top5_similarity']:<12.4f}"
                )

    # Save full comparison
    save_json(
        comparison_results,
        str(PROJECT_ROOT / "results/embedding_models_comparison.json")
    )

    print(f"\n\nResults saved to: results/embedding_models_comparison.json")


if __name__ == "__main__":
    main()
