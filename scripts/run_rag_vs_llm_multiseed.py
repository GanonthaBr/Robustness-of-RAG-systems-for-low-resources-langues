#!/usr/bin/env python3
"""Run RAG vs LLM-only across multiple seeds and aggregate metrics."""

import argparse
import os
import statistics
import sys

if sys.version_info[0] < 3:
    raise RuntimeError(
        "run_rag_vs_llm_multiseed.py requires Python 3. "
        "Run with 'python3 scripts/run_rag_vs_llm_multiseed.py'."
    )

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

from scripts.run_rag_vs_llm_once import main as run_once
from config.settings import LLM_MODELS, RAG_K_BY_LANGUAGE
from utils.helpers import save_json


def _mean(values):
    return float(statistics.mean(values)) if values else 0.0


def _std(values):
    if len(values) < 2:
        return 0.0
    return float(statistics.pstdev(values))


def _collect_lang_metric(seed_results, language, path):
    values = []
    for result in seed_results:
        node = result["by_language"][language]
        for key in path:
            node = node[key]
        values.append(float(node))
    return values


def main(num_examples, seeds, use_language_k=False, llm_model='afriqueqwen-8b'):
    languages = ["swa", "yor", "kin"]
    all_seed_results = []
    k_by_language = RAG_K_BY_LANGUAGE if use_language_k else None

    print("=" * 84)
    print("MULTI-SEED RAG VS LLM-ONLY (E5)")
    print("=" * 84)
    print("Seeds: {}".format(seeds))
    print("Examples per language: {}".format(num_examples))
    print("LLM model: {}".format(llm_model))
    if use_language_k:
        print("Language-specific k enabled: {}".format(k_by_language))

    for seed in seeds:
        print("\n" + "#" * 84)
        print("Running seed {}".format(seed))
        print("#" * 84)

        if use_language_k:
            per_seed_file = "results/rag_vs_llm_only_e5_tuned_{}_seed{}.json".format(num_examples, seed)
        else:
            per_seed_file = "results/rag_vs_llm_only_e5_{}_seed{}.json".format(num_examples, seed)

        result = run_once(
            num_examples=num_examples,
            seed=seed,
            output_file=per_seed_file,
            k_by_language=k_by_language,
            llm_model=llm_model,
        )
        all_seed_results.append(result)

    aggregated = {
        "metadata": {
            "num_examples": num_examples,
            "seeds": seeds,
            "num_seeds": len(seeds),
            "rag_embedding": "e5-base",
            "llm_model": llm_model,
            "k_by_language": k_by_language or {lang: 10 for lang in languages},
        },
        "by_language": {},
    }

    print("\n" + "=" * 84)
    print("AGGREGATED SUMMARY (MEAN +- STD)")
    print("=" * 84)

    for language in languages:
        delta_correct = _collect_lang_metric(
            all_seed_results, language, ["delta_rag_minus_llm_only", "correct_rate"]
        )
        rag_correct = _collect_lang_metric(
            all_seed_results, language, ["rag", "generation_metrics", "correct_rate"]
        )
        llm_correct = _collect_lang_metric(
            all_seed_results, language, ["llm_only", "generation_metrics", "correct_rate"]
        )
        rag_sim = _collect_lang_metric(
            all_seed_results, language, ["rag", "retriever_metrics", "mean_similarity"]
        )

        aggregated["by_language"][language] = {
            "rag_correct_rate": {
                "mean": _mean(rag_correct),
                "std": _std(rag_correct),
                "values": rag_correct,
            },
            "llm_only_correct_rate": {
                "mean": _mean(llm_correct),
                "std": _std(llm_correct),
                "values": llm_correct,
            },
            "delta_rag_minus_llm_only_correct_rate": {
                "mean": _mean(delta_correct),
                "std": _std(delta_correct),
                "values": delta_correct,
            },
            "rag_mean_similarity": {
                "mean": _mean(rag_sim),
                "std": _std(rag_sim),
                "values": rag_sim,
            },
        }

        print("\n{}:".format(language.upper()))
        print(
            "  RAG correct:      {:.1%} +- {:.1%}".format(
                aggregated["by_language"][language]["rag_correct_rate"]["mean"],
                aggregated["by_language"][language]["rag_correct_rate"]["std"],
            )
        )
        print(
            "  LLM-only correct: {:.1%} +- {:.1%}".format(
                aggregated["by_language"][language]["llm_only_correct_rate"]["mean"],
                aggregated["by_language"][language]["llm_only_correct_rate"]["std"],
            )
        )
        print(
            "  Delta (RAG-LLM):  {:+.1%} +- {:.1%}".format(
                aggregated["by_language"][language]["delta_rag_minus_llm_only_correct_rate"]["mean"],
                aggregated["by_language"][language]["delta_rag_minus_llm_only_correct_rate"]["std"],
            )
        )

    if use_language_k:
        output_file = PROJECT_ROOT / "results/rag_vs_llm_only_e5_tuned_multiseed_summary.json"
    else:
        output_file = PROJECT_ROOT / "results/rag_vs_llm_only_e5_multiseed_summary.json"
    save_json(aggregated, str(output_file))
    print("\nSaved aggregated results: {}".format(output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG vs LLM-only over multiple seeds")
    parser.add_argument("--num-examples", type=int, default=100, help="Examples per language")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="List of deterministic seeds",
    )
    parser.add_argument(
        "--use-language-k",
        action="store_true",
        help="Use tuned per-language k from config.RAG_K_BY_LANGUAGE",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        choices=list(LLM_MODELS.keys()),
        default='afriqueqwen-8b',
        help="LLM model key from config.LLM_MODELS",
    )
    args = parser.parse_args()

    main(num_examples=args.num_examples, seeds=args.seeds, use_language_k=args.use_language_k, llm_model=args.llm_model)
