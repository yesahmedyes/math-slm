"""Method 3: Self-Consistent Chain-of-Thought evaluation (majority voting).

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_sc_cot.py --model 4B --dataset gsm8k
"""

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from tqdm import tqdm

from config import MODELS, SAMPLING_CONFIGS
from inference import load_model
from data_loader import load_dataset_by_name
from prompts import SC_COT_SYSTEM, SC_COT_USER, build_messages
from answer_utils import extract_answer_from_model, compare_answers, normalize_answer

METHOD = "sc_cot"


def majority_vote(answers: list[str]) -> str:
    """Select the most common answer via majority voting."""
    normalized = []
    for a in answers:
        n = normalize_answer(a)
        if n is not None:
            normalized.append(n)
    if not normalized:
        return None
    counter = Counter(normalized)
    return counter.most_common(1)[0][0]


def evaluate(model, data, batch_size=32):
    cfg = SAMPLING_CONFIGS[METHOD]
    n_samples = cfg["n"]
    results = []

    all_prompts = []
    for ex in data:
        messages = build_messages(SC_COT_SYSTEM, SC_COT_USER, ex["problem"])
        prompt = model.build_prompt(messages, enable_thinking=False)
        all_prompts.append(prompt)

    all_outputs = []
    for i in tqdm(range(0, len(all_prompts), batch_size), desc="Generating"):
        batch = all_prompts[i:i + batch_size]
        # n > 1: each prompt gets n_samples completions
        outputs = model.generate(
            batch,
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            max_tokens=cfg["max_tokens"],
            n=n_samples,
        )
        all_outputs.extend(outputs)

    for ex, sample_outputs in zip(data, all_outputs):
        # sample_outputs is a list of n_samples strings
        individual_answers = [extract_answer_from_model(o) for o in sample_outputs]
        voted_answer = majority_vote(individual_answers)
        correct = compare_answers(voted_answer, ex["gold"], ex["source"])

        total_tokens = sum(model.count_tokens(o) for o in sample_outputs)

        results.append({
            "idx": ex.get("idx"),
            "problem": ex["problem"],
            "gold_answer": ex["gold"],
            "predicted_answer": voted_answer,
            "individual_answers": individual_answers,
            "correct": correct,
            "raw_outputs": sample_outputs,
            "output_tokens": total_tokens,
            "n_samples": n_samples,
        })

    return results


def compute_summary(results):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    avg_tokens = sum(r["output_tokens"] for r in results) / max(total, 1)
    return {
        "accuracy": correct / max(total, 1),
        "total": total,
        "correct": correct,
        "avg_tokens": round(avg_tokens, 1),
        "n_samples": results[0]["n_samples"] if results else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Self-Consistent CoT evaluation")
    parser.add_argument("--model", choices=list(MODELS.keys()), required=True)
    parser.add_argument("--dataset", required=True,
                        choices=["gsm8k", "math_algebra", "math_number_theory",
                                 "math_counting_prob", "all"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    model = load_model(args.model)

    if args.dataset == "all":
        datasets = load_dataset_by_name("all", args.max_samples)
    else:
        datasets = {args.dataset: load_dataset_by_name(args.dataset, args.max_samples)}

    os.makedirs(args.output_dir, exist_ok=True)

    for ds_name, data in datasets.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {METHOD} | model={args.model} | dataset={ds_name}")
        print(f"{'='*60}")

        results = evaluate(model, data, args.batch_size)
        summary = compute_summary(results)

        print(f"Accuracy: {summary['accuracy']:.4f} ({summary['correct']}/{summary['total']})")
        print(f"Avg tokens: {summary['avg_tokens']} (across {summary['n_samples']} samples)")

        output = {
            "method": METHOD,
            "model": MODELS[args.model],
            "dataset": ds_name,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "samples": results,
        }

        fname = f"{METHOD}_{args.model}_{ds_name}.json"
        path = os.path.join(args.output_dir, fname)
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {path}")


if __name__ == "__main__":
    main()
