"""Method 2: Unconstrained Chain-of-Thought evaluation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_cot.py --model 4B --dataset gsm8k
"""

import argparse
import json
import os
from datetime import datetime
from tqdm import tqdm

from config import MODELS, SAMPLING_CONFIGS
from inference import load_model
from data_loader import load_dataset_by_name
from prompts import COT_SYSTEM, COT_USER, build_messages
from answer_utils import extract_answer_from_model, compare_answers

METHOD = "cot"


def evaluate(model, data, batch_size=32):
    cfg = SAMPLING_CONFIGS[METHOD]
    results = []

    all_prompts = []
    for ex in data:
        messages = build_messages(COT_SYSTEM, COT_USER, ex["problem"])
        prompt = model.build_prompt(messages, enable_thinking=False)
        all_prompts.append(prompt)

    all_outputs = []
    for i in tqdm(range(0, len(all_prompts), batch_size), desc="Generating"):
        batch = all_prompts[i : i + batch_size]
        outputs = model.generate(
            batch,
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            max_tokens=cfg["max_tokens"],
        )
        all_outputs.extend(outputs)

    for ex, output in zip(data, all_outputs):
        predicted = extract_answer_from_model(output)
        correct = compare_answers(predicted, ex["gold"], ex["source"])
        results.append(
            {
                "idx": ex.get("idx"),
                "problem": ex["problem"],
                "gold_answer": ex["gold"],
                "predicted_answer": predicted,
                "correct": correct,
                "raw_output": output,
                "output_tokens": model.count_tokens(output),
            }
        )

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
    }


def main():
    parser = argparse.ArgumentParser(description="CoT evaluation")
    parser.add_argument("--model", choices=list(MODELS.keys()), required=True)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "gsm8k",
            "math_algebra",
            "math_number_theory",
            "math_counting_prob",
            "all",
        ],
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    model = load_model(args.model)

    if args.dataset == "all":
        datasets = load_dataset_by_name("all", args.max_samples)
    else:
        datasets = {args.dataset: load_dataset_by_name(args.dataset, args.max_samples)}

    os.makedirs(args.output_dir, exist_ok=True)

    for ds_name, data in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating {METHOD} | model={args.model} | dataset={ds_name}")
        print(f"{'=' * 60}")

        results = evaluate(model, data, args.batch_size)
        summary = compute_summary(results)

        print(
            f"Accuracy: {summary['accuracy']:.4f} ({summary['correct']}/{summary['total']})"
        )
        print(f"Avg tokens: {summary['avg_tokens']}")

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
