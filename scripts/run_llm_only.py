#!/usr/bin/env python3
"""Run LLM-only baseline (no retrieval) for language comparison."""

import argparse
import os
import sys

if sys.version_info[0] < 3:
    raise RuntimeError(
        "run_llm_only.py requires Python 3. Run with 'python3 run_llm_only.py --num-examples 10' "
        "or activate your venv before running."
    )

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
from evaluation.metrics import Evaluator
from pipeline.rag_pipeline import RAGPipeline
from config.settings import LLM_MODELS
from utils.helpers import save_json

load_dotenv(PROJECT_ROOT / ".env")


def main(num_examples=10, llm_model='afriqueqwen-8b'):
    """Run LLM-only experiments across supported African languages."""
    languages = ["swa", "yor", "kin"]
    all_results = {}

    print("AFRI-RAG LLM-ONLY BASELINE")
    print("Languages: {}".format(languages))
    print("Examples per language: {}".format(num_examples))
    print("LLM model: {}".format(llm_model))

    for language in languages:
        print("\n")
        print("Language: {}".format(language))

        loader = AfriQALoader()
        examples = loader.load(language, split="test", num_samples=num_examples)

        # Disable retrieval to test pure LLM ability.
        pipeline = RAGPipeline(language, use_retrieval=False, llm_model=llm_model)
        evaluator = Evaluator(language=language)

        predictions = []
        golds_local = []
        golds_en = []

        for i, ex in enumerate(examples):
            print("\nExample {}/{}".format(i + 1, len(examples)))
            print("Q: {}".format(ex['question']))

            result = pipeline.run(ex["question"], return_docs=False)
            predictions.append(result["answer"])

            answers = ex.get("answers", "")
            if isinstance(answers, list):
                golds_local.append(answers[0] if answers else "")
            else:
                golds_local.append(answers)

            golds_en.append(ex.get("translated_answer", ""))

        results = evaluator.evaluate_batch(predictions, golds_local, golds_en)
        evaluator.print_summary(results)

        all_results[language] = {
            "mode": "llm_only",
            "retriever_metrics": {
                "mean_similarity": 0.0,
                "mean_top1_similarity": 0.0,
                "mean_top5_similarity": 0.0,
                "max_similarity": 0.0,
                "min_similarity": 0.0,
                "num_queries": len(examples),
            },
            "generation_metrics": {
                "contains_gold_local": results["contains_gold_local"],
                "contains_gold_english": results["contains_gold_english"],
                "abstention_rate": results["abstention_rate"],
                "correct_rate": results["correct_rate"],
                "precision_on_answered": results["precision_on_answered"],
                "num_samples": results["num_samples"],
                "num_abstained": results["num_abstained"],
            },
        }

        save_json(results, str(PROJECT_ROOT / "results/{}_llm_only_results.json".format(language)))

    print("\n")
    print("=" * 70)
    print("FINAL LLM-ONLY SUMMARY BY LANGUAGE")
    print("=" * 70)
    for language, lang_results in all_results.items():
        gen_metrics = lang_results.get("generation_metrics", {})
        print("\n{}:".format(language.upper()))
        print(
            "  Generation: correct={:.1%}, local={:.1%}, en={:.1%}, abstain={:.1%}".format(
                gen_metrics.get('correct_rate', 0),
                gen_metrics.get('contains_gold_local', 0),
                gen_metrics.get('contains_gold_english', 0),
                gen_metrics.get('abstention_rate', 0),
            )
        )

    save_json(all_results, str(PROJECT_ROOT / "results/all_languages_llm_only_summary.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM-only baseline without retrieval")
    parser.add_argument("--num-examples", type=int, default=10, help="Number of test examples per language")
    parser.add_argument(
        "--llm-model",
        type=str,
        choices=list(LLM_MODELS.keys()),
        default='afriqueqwen-8b',
        help="LLM model key from config.LLM_MODELS",
    )
    args = parser.parse_args()
    main(num_examples=args.num_examples, llm_model=args.llm_model)
