"""Method 7: Logic-LM (LLM + Symbolic Solver with Self-Refinement) evaluation.

Three-stage framework adapted from Pan et al. (2023) "Logic-LM: Empowering
Large Language Models with Symbolic Solvers for Faithful Logical Reasoning":
  1. Problem Formulator: LLM translates math problem into structured SymPy code
  2. Symbolic Reasoner:  Execute the SymPy program deterministically
  3. Self-Refinement:   If execution fails, feed error back to LLM to fix the
                        formulation (up to max_refine rounds)

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_logic_lm.py --model 4B --dataset gsm8k
"""

import argparse
import json
import os
from datetime import datetime
from tqdm import tqdm

from config import MODELS, SAMPLING_CONFIGS
from inference import load_model
from data_loader import load_dataset_by_name
from prompts import (
    LOGIC_LM_SYSTEM,
    LOGIC_LM_USER,
    LOGIC_LM_REFINE_SYSTEM,
    LOGIC_LM_REFINE_USER,
    build_messages,
)
from answer_utils import extract_answer_from_model, compare_answers
from sandbox import extract_code_from_output, execute_code

METHOD = "logic_lm"


def formulate_and_solve(model, problem, cfg):
    """Stage 1+2: Problem Formulation → Symbolic Reasoning.

    Returns (output, code, exec_result).
    """
    messages = build_messages(LOGIC_LM_SYSTEM, LOGIC_LM_USER, problem)
    prompt = model.build_prompt(messages, enable_thinking=False)
    outputs = model.generate(
        [prompt],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        max_tokens=cfg["max_tokens"],
    )
    output = outputs[0]
    code = extract_code_from_output(output)
    exec_result = execute_code(code)
    return output, code, exec_result


def self_refine(model, problem, code, error, cfg):
    """Stage 3: Self-Refinement — feed solver error back to LLM.

    Returns (output, new_code, exec_result).
    """
    user_content = LOGIC_LM_REFINE_USER.format(problem=problem, code=code, error=error)
    messages = [
        {"role": "system", "content": LOGIC_LM_REFINE_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    prompt = model.build_prompt(messages, enable_thinking=False)
    outputs = model.generate(
        [prompt],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        max_tokens=cfg["max_tokens"],
    )
    output = outputs[0]
    new_code = extract_code_from_output(output)
    exec_result = execute_code(new_code)
    return output, new_code, exec_result


def evaluate(model, data, batch_size=1):
    """Evaluate Logic-LM on a dataset.

    Note: batch_size is effectively 1 due to the sequential self-refinement
    loop. The initial formulation is batched, but refinement is per-sample.
    """
    cfg = SAMPLING_CONFIGS[METHOD]
    max_refine = cfg.get("max_refine", 3)
    results = []

    # --- Stage 1: Batch the initial formulation ---
    all_prompts = []
    for ex in data:
        messages = build_messages(LOGIC_LM_SYSTEM, LOGIC_LM_USER, ex["problem"])
        prompt = model.build_prompt(messages, enable_thinking=False)
        all_prompts.append(prompt)

    all_initial_outputs = []
    for i in tqdm(range(0, len(all_prompts), batch_size), desc="Formulating"):
        batch = all_prompts[i : i + batch_size]
        outputs = model.generate(
            batch,
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            max_tokens=cfg["max_tokens"],
        )
        all_initial_outputs.extend(outputs)

    # --- Stage 2+3: Execute and Self-Refine per sample ---
    for ex, output in tqdm(
        zip(data, all_initial_outputs), total=len(data), desc="Solving+Refining"
    ):
        code = extract_code_from_output(output)
        exec_result = execute_code(code)

        refinement_rounds = 0
        all_outputs = [output]
        all_codes = [code]
        all_errors = []

        # Self-refinement loop (Logic-LM's key innovation)
        while not exec_result["success"] and refinement_rounds < max_refine:
            error_msg = exec_result.get(
                "error", exec_result.get("stderr", "Unknown error")
            )
            all_errors.append(error_msg)

            refine_output, new_code, exec_result = self_refine(
                model, ex["problem"], code, error_msg, cfg
            )
            refinement_rounds += 1
            code = new_code
            output = refine_output
            all_outputs.append(output)
            all_codes.append(code)

        # Extract answer
        if exec_result["success"] and exec_result["stdout"]:
            predicted = extract_answer_from_model(exec_result["stdout"])
        else:
            # Fallback: try extracting from raw LLM output (like CoT)
            predicted = extract_answer_from_model(output)

        correct = compare_answers(predicted, ex["gold"], ex["source"])

        total_tokens = sum(model.count_tokens(o) for o in all_outputs)

        results.append(
            {
                "idx": ex.get("idx"),
                "problem": ex["problem"],
                "gold_answer": ex["gold"],
                "predicted_answer": predicted,
                "correct": correct,
                "raw_output": all_outputs[0],
                "final_code": code,
                "exec_stdout": exec_result["stdout"],
                "exec_stderr": exec_result["stderr"],
                "exec_success": exec_result["success"],
                "refinement_rounds": refinement_rounds,
                "refinement_errors": all_errors,
                "output_tokens": total_tokens,
            }
        )

    return results


def compute_summary(results):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    exec_success = sum(1 for r in results if r["exec_success"])
    avg_tokens = sum(r["output_tokens"] for r in results) / max(total, 1)
    avg_refine = sum(r["refinement_rounds"] for r in results) / max(total, 1)
    needed_refine = sum(1 for r in results if r["refinement_rounds"] > 0)

    # Exe_Rate and Exe_Acc (from Logic-LM paper Table 3)
    exe_rate = exec_success / max(total, 1)  # % of executable formulations
    exe_acc = sum(1 for r in results if r["exec_success"] and r["correct"]) / max(
        exec_success, 1
    )  # accuracy among executable samples

    return {
        "accuracy": correct / max(total, 1),
        "total": total,
        "correct": correct,
        "exec_success_rate": exe_rate,
        "exec_accuracy": round(exe_acc, 4),
        "avg_refinement_rounds": round(avg_refine, 2),
        "samples_needing_refinement": needed_refine,
        "avg_tokens": round(avg_tokens, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Logic-LM evaluation")
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
        print(f"Code exec success (Exe_Rate): {summary['exec_success_rate']:.4f}")
        print(f"Exec accuracy (Exe_Acc): {summary['exec_accuracy']:.4f}")
        print(
            f"Self-refinement: {summary['samples_needing_refinement']}/{summary['total']} "
            f"samples, avg {summary['avg_refinement_rounds']} rounds"
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
