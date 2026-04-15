#!/usr/bin/env python3
"""Run RAG k-sweep vs LLM-only across multiple seeds."""

import argparse
import os
import random
import sys

if sys.version_info[0] < 3:
    raise RuntimeError(
        "run_rag_k_sweep_multiseed.py requires Python 3. "
        "Run with 'python3 scripts/run_rag_k_sweep_multiseed.py'."
    )

import statistics

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

from config.settings import EMBEDDING_MODELS, LLM_MODELS
from data.dataset import AfriQALoader
from evaluation.metrics import Evaluator, RetrieverEvaluator
from pipeline.rag_pipeline import RAGPipeline
from utils.helpers import save_json

load_dotenv(PROJECT_ROOT / ".env")


def _sample_examples(examples, num_examples, seed):
    if num_examples >= len(examples):
        return examples
    rng = random.Random(seed)
    indices = rng.sample(range(len(examples)), num_examples)
    return [examples[i] for i in indices]


def _mean(values):
    return float(statistics.mean(values)) if values else 0.0


def _std(values):
    if len(values) < 2:
        return 0.0
    return float(statistics.pstdev(values))


def _evaluate_mode(pipeline, examples, evaluator, retrieval_k=None, capture_docs=False):
    predictions = []
    golds_local = []
    golds_en = []
    all_docs = []

    for i, ex in enumerate(examples):
        print("    Example {}/{}".format(i + 1, len(examples)))
        result = pipeline.run(ex["question"], k=retrieval_k or 10, return_docs=capture_docs)
        predictions.append(result["answer"])

        answers = ex.get("answers", "")
        if isinstance(answers, list):
            golds_local.append(answers[0] if answers else "")
        else:
            golds_local.append(answers)

        golds_en.append(ex.get("translated_answer", ""))

        docs = result.get("documents", []) if capture_docs else []
        all_docs.append(docs)

    gen_metrics = evaluator.evaluate_batch(predictions, golds_local, golds_en)
    ret_metrics = RetrieverEvaluator.evaluate_retrieval(all_docs) if capture_docs else {
        "mean_similarity": 0.0,
        "mean_top1_similarity": 0.0,
        "mean_top5_similarity": 0.0,
        "max_similarity": 0.0,
        "min_similarity": 0.0,
        "num_queries": len(examples),
    }

    return {
        "generation_metrics": gen_metrics,
        "retriever_metrics": ret_metrics,
    }


def main(num_examples, seeds, k_values, llm_model='afriqueqwen-8b'):
    languages = ["swa", "yor", "kin"]
    e5_model = EMBEDDING_MODELS.get("e5-base", "intfloat/multilingual-e5-base")
    llm_tag = str(llm_model).replace("/", "-").replace(":", "-")

    print("=" * 88)
    print("RAG K-SWEEP VS LLM-ONLY (MULTI-SEED, E5)")
    print("=" * 88)
    print("Languages: {}".format(languages))
    print("Examples per language: {}".format(num_examples))
    print("Seeds: {}".format(seeds))
    print("K values: {}".format(k_values))
    print("RAG embedding: e5-base ({})".format(e5_model))
    print("LLM model: {}".format(llm_model))

    loader = AfriQALoader()
    per_seed = []

    for seed in seeds:
        print("\n" + "#" * 88)
        print("Running seed {}".format(seed))
        print("#" * 88)

        seed_result = {
            "seed": seed,
            "by_language": {},
        }

        for language in languages:
            print("\n" + "-" * 88)
            print("Language: {}".format(language))
            print("-" * 88)

            all_examples = loader.load(language, split="test", num_samples=None)
            examples = _sample_examples(all_examples, num_examples, seed)

            llm_pipeline = RAGPipeline(language, use_retrieval=False, llm_model=llm_model)
            llm_eval = Evaluator(language=language)
            print("  Running LLM-only baseline...")
            llm_metrics = _evaluate_mode(llm_pipeline, examples, llm_eval, retrieval_k=None, capture_docs=False)

            rag_pipeline = RAGPipeline(language, use_retrieval=True, embedding_model=e5_model, llm_model=llm_model)
            rag_eval = Evaluator(language=language)

            rag_by_k = {}
            for k in k_values:
                print("  Running RAG@{}...".format(k))
                rag_metrics = _evaluate_mode(rag_pipeline, examples, rag_eval, retrieval_k=k, capture_docs=True)

                delta = {
                    "correct_rate": rag_metrics["generation_metrics"].get("correct_rate", 0.0)
                    - llm_metrics["generation_metrics"].get("correct_rate", 0.0),
                    "contains_gold_local": rag_metrics["generation_metrics"].get("contains_gold_local", 0.0)
                    - llm_metrics["generation_metrics"].get("contains_gold_local", 0.0),
                    "contains_gold_english": rag_metrics["generation_metrics"].get("contains_gold_english", 0.0)
                    - llm_metrics["generation_metrics"].get("contains_gold_english", 0.0),
                    "abstention_rate": rag_metrics["generation_metrics"].get("abstention_rate", 0.0)
                    - llm_metrics["generation_metrics"].get("abstention_rate", 0.0),
                }

                rag_by_k[str(k)] = {
                    "rag": rag_metrics,
                    "delta_rag_minus_llm_only": delta,
                }

            seed_result["by_language"][language] = {
                "llm_only": llm_metrics,
                "rag_by_k": rag_by_k,
            }

        per_seed.append(seed_result)
        save_json(seed_result, str(PROJECT_ROOT / "results/rag_k_sweep_{}_seed{}.json".format(llm_tag, seed)))

    aggregated = {
        "metadata": {
            "num_examples": num_examples,
            "seeds": seeds,
            "num_seeds": len(seeds),
            "k_values": k_values,
            "rag_embedding": "e5-base",
            "rag_embedding_model": e5_model,
            "llm_model": llm_model,
        },
        "by_language": {},
    }

    print("\n" + "=" * 88)
    print("AGGREGATED SUMMARY (MEAN +- STD)")
    print("=" * 88)

    for language in languages:
        aggregated["by_language"][language] = {
            "llm_only_correct_rate": {},
            "rag_by_k": {},
        }

        llm_vals = [
            seed_item["by_language"][language]["llm_only"]["generation_metrics"].get("correct_rate", 0.0)
            for seed_item in per_seed
        ]
        aggregated["by_language"][language]["llm_only_correct_rate"] = {
            "mean": _mean(llm_vals),
            "std": _std(llm_vals),
            "values": llm_vals,
        }

        print("\n{}:".format(language.upper()))
        print(
            "  LLM-only correct: {:.1%} +- {:.1%}".format(
                aggregated["by_language"][language]["llm_only_correct_rate"]["mean"],
                aggregated["by_language"][language]["llm_only_correct_rate"]["std"],
            )
        )

        for k in k_values:
            k_key = str(k)
            rag_correct_vals = [
                seed_item["by_language"][language]["rag_by_k"][k_key]["rag"]["generation_metrics"].get("correct_rate", 0.0)
                for seed_item in per_seed
            ]
            delta_vals = [
                seed_item["by_language"][language]["rag_by_k"][k_key]["delta_rag_minus_llm_only"].get("correct_rate", 0.0)
                for seed_item in per_seed
            ]

            aggregated["by_language"][language]["rag_by_k"][k_key] = {
                "rag_correct_rate": {
                    "mean": _mean(rag_correct_vals),
                    "std": _std(rag_correct_vals),
                    "values": rag_correct_vals,
                },
                "delta_rag_minus_llm_only_correct_rate": {
                    "mean": _mean(delta_vals),
                    "std": _std(delta_vals),
                    "values": delta_vals,
                },
            }

            print(
                "  RAG@{} correct: {:.1%} +- {:.1%} | Delta: {:+.1%} +- {:.1%}".format(
                    k,
                    aggregated["by_language"][language]["rag_by_k"][k_key]["rag_correct_rate"]["mean"],
                    aggregated["by_language"][language]["rag_by_k"][k_key]["rag_correct_rate"]["std"],
                    aggregated["by_language"][language]["rag_by_k"][k_key]["delta_rag_minus_llm_only_correct_rate"]["mean"],
                    aggregated["by_language"][language]["rag_by_k"][k_key]["delta_rag_minus_llm_only_correct_rate"]["std"],
                )
            )

    output_file = PROJECT_ROOT / "results/rag_k_sweep_{}_multiseed_summary.json".format(llm_tag)
    save_json(aggregated, str(output_file))
    print("\nSaved aggregated summary: {}".format(output_file))
    return str(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG k-sweep vs LLM-only over multiple seeds")
    parser.add_argument("--num-examples", type=int, default=100, help="Examples per language")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="List of deterministic seeds",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[3, 5, 10],
        help="RAG retrieval k values to test",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        choices=list(LLM_MODELS.keys()),
        default='afriqueqwen-8b',
        help="LLM model key from config.LLM_MODELS",
    )
    parser.add_argument(
        "--all-llms",
        action="store_true",
        help="Run the full experiment for every model listed in config.LLM_MODELS",
    )
    args = parser.parse_args()

    if args.all_llms:
        combined = {
            "metadata": {
                "num_examples": args.num_examples,
                "seeds": args.seeds,
                "k_values": args.k_values,
                "llm_models": list(LLM_MODELS.keys()),
            },
            "outputs": {},
        }

        for llm_key in LLM_MODELS.keys():
            print("\n" + "=" * 96)
            print("RUNNING MODEL: {}".format(llm_key))
            print("=" * 96)
            out_path = main(
                num_examples=args.num_examples,
                seeds=args.seeds,
                k_values=args.k_values,
                llm_model=llm_key,
            )
            combined["outputs"][llm_key] = out_path

        combined_file = PROJECT_ROOT / "results/rag_k_sweep_all_llms_multiseed_index.json"
        save_json(combined, str(combined_file))
        print("\nSaved all-LLM index: {}".format(combined_file))
    else:
        main(num_examples=args.num_examples, seeds=args.seeds, k_values=args.k_values, llm_model=args.llm_model)
