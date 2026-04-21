#!/usr/bin/env python3
"""Run robustness experiments with checkpointing and progress monitoring."""

import argparse
import copy
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

if sys.version_info[0] < 3:
    raise RuntimeError(
        "run_robustness_multiseed.py requires Python 3. "
        "Run with 'python3 scripts/run_robustness_multiseed.py'."
    )

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

from config.settings import EMBEDDING_MODELS, LLM_MODELS, LLM_MODEL
from data.dataset import AfriQALoader
from data.wikipedia import WikipediaCorpus
from evaluation.metrics import Evaluator
from generation.afrique_qwen import AfriqueQwenGenerator
from generation.prompts import PromptManager
from retrieval.dense_retriever import DenseRetriever
from utils.helpers import load_json, save_json

load_dotenv(PROJECT_ROOT / ".env")

# Best-k from completed clean comparison runs.
BEST_K_BY_MODEL = {
    "afriqueqwen-8b": {"swa": 10, "yor": 3, "kin": 10},
    "qwen2.5-7b-instruct": {"swa": 3, "yor": 3, "kin": 10},
}

NOISE_TYPES = [
    "irrelevant",
    "contradictory",
    "partially_correct",
    "translation_artifact",
]
SEVERITIES = [10, 30, 50]


class ProgressTracker:
    """Track run progress with ETA."""

    def __init__(self, total_units):
        self.total_units = max(1, int(total_units))
        self.done_units = 0
        self.start_time = time.time()

    @staticmethod
    def _fmt_seconds(seconds):
        seconds = max(0, int(seconds))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return "{:02d}:{:02d}:{:02d}".format(hours, minutes, secs)

    def update(self, amount=1, label="", force=False):
        self.done_units += int(amount)
        if not force and self.done_units % 50 != 0:
            return

        elapsed = time.time() - self.start_time
        rate = self.done_units / elapsed if elapsed > 0 else 0.0
        remaining = self.total_units - self.done_units
        eta_seconds = remaining / rate if rate > 0 else 0.0
        pct = 100.0 * self.done_units / self.total_units

        print(
            "[Progress] {:6.2f}% ({}/{}) | elapsed={} | eta={}{}".format(
                pct,
                self.done_units,
                self.total_units,
                self._fmt_seconds(elapsed),
                self._fmt_seconds(eta_seconds),
                " | {}".format(label) if label else "",
            )
        )


def _model_tag(model_key):
    return str(model_key).replace("/", "-").replace(":", "-")


def _resolve_model_name(model_key):
    if model_key in LLM_MODELS:
        return LLM_MODELS[model_key]
    if model_key:
        return model_key
    return LLM_MODEL


def _build_conditions():
    conditions = [{"name": "llm_only", "noise_type": None, "severity": 0}]
    conditions.append({"name": "clean", "noise_type": None, "severity": 0})
    for noise in NOISE_TYPES:
        for severity in SEVERITIES:
            conditions.append(
                {
                    "name": "{}_{}".format(noise, severity),
                    "noise_type": noise,
                    "severity": severity,
                }
            )
    return conditions


def _sample_examples(examples, num_examples, seed):
    if num_examples >= len(examples):
        return examples
    rng = random.Random(seed)
    indices = rng.sample(range(len(examples)), num_examples)
    return [examples[i] for i in indices]


def _translation_artifact_text(text):
    # Deterministic, lightweight synthetic corruption.
    repl = {
        "a": "", "e": "", "i": "", "o": "", "u": "",
        "A": "", "E": "", "I": "", "O": "", "U": "",
    }
    out = []
    for i, ch in enumerate(text):
        if i % 17 == 0 and ch.isalpha():
            continue
        out.append(repl.get(ch, ch))
    return "".join(out).strip() or text


def _contradiction_prefix(language):
    prefixes = {
        "swa": "TAHADHARI: Maelezo yafuatayo yanaweza kupingana na ukweli.",
        "yor": "IKILO: Alaye yi le tako otito.",
        "kin": "IYITONDE: Ibi bikurikira bishobora kunyuranya n'ukuri.",
        "en": "WARNING: The following statement may contradict factual evidence.",
    }
    return prefixes.get(language, prefixes["en"])


def _apply_noise(docs, noise_type, severity, language, passage_pool, rng):
    if noise_type is None or severity <= 0:
        return copy.deepcopy(docs)

    noisy = copy.deepcopy(docs)
    if not noisy:
        return noisy

    count = max(1, int(round(len(noisy) * (float(severity) / 100.0))))
    idxs = rng.sample(range(len(noisy)), min(count, len(noisy)))

    if noise_type == "irrelevant":
        for i in idxs:
            replacement = rng.choice(passage_pool)
            noisy[i]["text"] = replacement.get("text", "")
            noisy[i]["score"] = 0.0
            noisy[i]["noise_flag"] = "irrelevant"
        return noisy

    if noise_type == "contradictory":
        prefix = _contradiction_prefix(language)
        for i in idxs:
            noisy[i]["text"] = "{} {}".format(prefix, noisy[i].get("text", ""))
            noisy[i]["noise_flag"] = "contradictory"
        return noisy

    if noise_type == "partially_correct":
        for i in idxs:
            words = noisy[i].get("text", "").split()
            keep = max(5, int(len(words) * 0.35))
            noisy[i]["text"] = " ".join(words[:keep])
            noisy[i]["noise_flag"] = "partially_correct"
        return noisy

    if noise_type == "translation_artifact":
        for i in idxs:
            noisy[i]["text"] = _translation_artifact_text(noisy[i].get("text", ""))
            noisy[i]["noise_flag"] = "translation_artifact"
        return noisy

    return noisy


def _ensure_state(state_path):
    if state_path.exists():
        return load_json(str(state_path))
    return {"completed_blocks": []}


def _save_state(state_path, state):
    save_json(state, str(state_path))


def _mean(values):
    return float(statistics.mean(values)) if values else 0.0


def _std(values):
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _build_golds(examples):
    golds_local = []
    golds_en = []
    for ex in examples:
        answers = ex.get("answers", "")
        if isinstance(answers, list):
            golds_local.append(answers[0] if answers else "")
        else:
            golds_local.append(answers)
        golds_en.append(ex.get("translated_answer", ""))
    return golds_local, golds_en


def _generate_for_condition(
    generator,
    prompt_manager,
    examples,
    retrieved_docs,
    condition,
    language,
    passage_pool,
    progress,
    rng,
    max_new_tokens,
    temperature,
):
    predictions = []
    cond_name = condition["name"]

    for i, ex in enumerate(examples):
        if cond_name == "llm_only":
            prompt = prompt_manager.create_prompt(
                question=ex["question"],
                documents=[],
                include_docs=False,
            )
        else:
            docs = retrieved_docs[i]
            noisy_docs = _apply_noise(
                docs=docs,
                noise_type=condition["noise_type"],
                severity=condition["severity"],
                language=language,
                passage_pool=passage_pool,
                rng=rng,
            )
            prompt = prompt_manager.create_prompt(
                question=ex["question"],
                documents=noisy_docs,
                include_docs=True,
            )

        result = generator.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        predictions.append(result.get("text", ""))
        progress.update(1, label="{} {} ex {}/{}".format(language, cond_name, i + 1, len(examples)))

    return predictions


def main(
    num_examples,
    seeds,
    languages,
    llm_models,
    embedding_model,
    max_new_tokens,
    temperature,
    resume,
):
    from config.settings import MAX_NEW_TOKENS, TEMPERATURE

    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS
    if temperature is None:
        temperature = TEMPERATURE

    conditions = _build_conditions()
    loader = AfriQALoader()

    output_root = PROJECT_ROOT / "results"
    checkpoint_root = output_root / "robustness_checkpoints"
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    state_path = checkpoint_root / "robustness_run_state.json"
    state = _ensure_state(state_path)
    completed = set(state.get("completed_blocks", []))

    total_units = len(llm_models) * len(languages) * len(seeds) * len(conditions) * num_examples
    progress = ProgressTracker(total_units=total_units)

    print("=" * 96)
    print("ROBUSTNESS MULTI-SEED RUN")
    print("=" * 96)
    print("Models: {}".format(llm_models))
    print("Languages: {}".format(languages))
    print("Seeds: {}".format(seeds))
    print("Conditions: {}".format([c["name"] for c in conditions]))
    print("Examples/language: {}".format(num_examples))
    print("Embedding model: {}".format(embedding_model))
    print("Max new tokens: {}".format(max_new_tokens))
    print("Temperature: {}".format(temperature))
    print("Resume mode: {}".format(resume))

    # Build retriever cache once per language for speed.
    retriever_by_language = {}
    passage_pool_by_language = {}
    for language in languages:
        print("\nPreparing retriever cache for {}...".format(language))
        corpus = WikipediaCorpus(language)
        passage_pool = corpus.get_passages()
        retriever = DenseRetriever(model_name=embedding_model)
        retriever.index_corpus(passage_pool)
        retriever_by_language[language] = retriever
        passage_pool_by_language[language] = passage_pool

    all_outputs = {}

    for llm_key in llm_models:
        model_name = _resolve_model_name(llm_key)
        llm_tag = _model_tag(llm_key)

        print("\n" + "#" * 96)
        print("MODEL: {} -> {}".format(llm_key, model_name))
        print("#" * 96)

        print("Loading generator once for model {}...".format(llm_key))
        generator = AfriqueQwenGenerator(model_name=model_name)

        per_model_records = []

        for language in languages:
            prompt_manager = PromptManager(language)
            evaluator = Evaluator(language=language)
            retriever = retriever_by_language[language]
            passage_pool = passage_pool_by_language[language]
            best_k = BEST_K_BY_MODEL.get(llm_key, {}).get(language, 5)

            print("\nLanguage {} | best_k={}".format(language, best_k))

            all_examples = loader.load(language, split="test", num_samples=None)

            for seed in seeds:
                seed_rng = random.Random(seed)
                examples = _sample_examples(all_examples, num_examples, seed)
                golds_local, golds_en = _build_golds(examples)

                # Retrieve once per example for all RAG conditions.
                retrieved_docs = []
                for ex in examples:
                    retrieved_docs.append(retriever.retrieve(ex["question"], k=int(best_k)))

                for condition in conditions:
                    cond_name = condition["name"]
                    block_id = "{}|{}|seed{}|{}".format(llm_key, language, seed, cond_name)
                    block_file = checkpoint_root / (
                        "robustness_{}_{}_seed{}_{}.json".format(llm_tag, language, seed, cond_name)
                    )

                    if resume and block_id in completed and block_file.exists():
                        print("Skipping completed block: {}".format(block_id))
                        data = load_json(str(block_file))
                        per_model_records.append(data)
                        progress.update(num_examples, label="skipped {}".format(block_id), force=True)
                        continue

                    print("Running {}".format(block_id))
                    predictions = _generate_for_condition(
                        generator=generator,
                        prompt_manager=prompt_manager,
                        examples=examples,
                        retrieved_docs=retrieved_docs,
                        condition=condition,
                        language=language,
                        passage_pool=passage_pool,
                        progress=progress,
                        rng=seed_rng,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )

                    metrics = evaluator.evaluate_batch(predictions, golds_local, golds_en)

                    record = {
                        "metadata": {
                            "llm_model": llm_key,
                            "llm_model_path": model_name,
                            "language": language,
                            "seed": seed,
                            "num_examples": len(examples),
                            "condition": cond_name,
                            "noise_type": condition["noise_type"],
                            "severity": condition["severity"],
                            "best_k": best_k,
                            "embedding_model": embedding_model,
                        },
                        "metrics": metrics,
                    }
                    save_json(record, str(block_file))
                    per_model_records.append(record)

                    completed.add(block_id)
                    state["completed_blocks"] = sorted(list(completed))
                    _save_state(state_path, state)
                    progress.update(0, label="completed {}".format(block_id), force=True)

        # Aggregate model summary
        model_summary = {
            "metadata": {
                "llm_model": llm_key,
                "llm_model_path": model_name,
                "languages": languages,
                "seeds": seeds,
                "num_examples": num_examples,
                "conditions": [c["name"] for c in conditions],
            },
            "by_language": {},
        }

        for language in languages:
            lang_records = [
                r for r in per_model_records if r["metadata"]["language"] == language
            ]
            by_condition = {}
            for condition in conditions:
                cname = condition["name"]
                cond_records = [
                    r for r in lang_records if r["metadata"]["condition"] == cname
                ]
                values = [float(r["metrics"].get("correct_rate", 0.0)) for r in cond_records]
                by_condition[cname] = {
                    "correct_rate": {
                        "mean": _mean(values),
                        "std": _std(values),
                        "values": values,
                    }
                }

            clean_mean = by_condition.get("clean", {}).get("correct_rate", {}).get("mean", 0.0)
            for cname, node in by_condition.items():
                cmean = node["correct_rate"]["mean"]
                node["delta_vs_clean_mean"] = float(cmean - clean_mean)

            model_summary["by_language"][language] = by_condition

        model_summary_file = output_root / "robustness_{}_multiseed_summary.json".format(llm_tag)
        save_json(model_summary, str(model_summary_file))
        all_outputs[llm_key] = str(model_summary_file)
        print("Saved model robustness summary: {}".format(model_summary_file))

    run_index = {
        "metadata": {
            "num_examples": num_examples,
            "seeds": seeds,
            "languages": languages,
            "llm_models": llm_models,
            "embedding_model": embedding_model,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
        "outputs": all_outputs,
        "checkpoint_state": str(state_path),
    }

    run_index_file = output_root / "robustness_all_llms_index.json"
    save_json(run_index, str(run_index_file))

    print("\nSaved run index: {}".format(run_index_file))
    progress.update(0, label="all done", force=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run robustness experiments with checkpointing")
    parser.add_argument("--num-examples", type=int, default=50, help="Examples per language")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="List of deterministic seeds",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["swa", "yor", "kin"],
        help="Languages to run",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        choices=list(LLM_MODELS.keys()),
        help="Single LLM model key from config.LLM_MODELS",
    )
    parser.add_argument(
        "--all-llms",
        action="store_true",
        help="Run all models from config.LLM_MODELS",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="e5-base",
        choices=list(EMBEDDING_MODELS.keys()),
        help="Embedding model key from config.EMBEDDING_MODELS",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override max generation tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override generation temperature",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint resume behavior",
    )

    args = parser.parse_args()

    if args.all_llms:
        model_keys = list(LLM_MODELS.keys())
    elif args.llm_model:
        model_keys = [args.llm_model]
    else:
        model_keys = [list(LLM_MODELS.keys())[0]]

    emb_model_path = EMBEDDING_MODELS.get(args.embedding_model, EMBEDDING_MODELS["e5-base"])

    main(
        num_examples=args.num_examples,
        seeds=args.seeds,
        languages=args.languages,
        llm_models=model_keys,
        embedding_model=emb_model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        resume=not args.no_resume,
    )
