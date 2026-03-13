"""Method X: S3-Math (Selective Neuro-Symbolic State Scaffolding) evaluation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_s3math.py --model 4B --dataset gsm8k
    CUDA_VISIBLE_DEVICES=1 python run_s3math.py --model 2B --dataset all --max_samples 100
"""

import argparse
import json
import os
from datetime import datetime
from tqdm import tqdm

from config import MODELS, SAMPLING_CONFIGS, S3MATH_MAX_REPAIRS
from inference import load_model
from data_loader import load_dataset_by_name
from prompts import (
    S3MATH_SYSTEM,
    S3MATH_USER,
    S3MATH_REPAIR,
    build_messages,
)
from answer_utils import extract_answer_from_model, compare_answers
from symbolic_engine import (
    SymbolicState,
    parse_trace,
    execute_trace,
    has_errors,
    format_trace_with_errors,
    format_error_summary,
    extract_answer_from_trace,
    SYMBOLIC_OPS,
)

METHOD = "s3math"


def solve_one(model, problem: str, max_repairs: int = S3MATH_MAX_REPAIRS) -> dict:
    """Solve a single problem using S3-Math with repair loop.

    Returns dict with answer, trace info, and diagnostics.
    """
    cfg = SAMPLING_CONFIGS[METHOD]

    # Initial generation
    messages = build_messages(S3MATH_SYSTEM, S3MATH_USER, problem)
    prompt = model.build_prompt(messages, enable_thinking=False)
    output = model.generate([prompt], **{k: v for k, v in cfg.items()})[0]

    total_tokens = model.count_tokens(output)
    all_outputs = [output]

    # Parse and execute
    steps = parse_trace(output)
    state = SymbolicState()

    if steps:
        answer, steps, stats = execute_trace(steps, state, SYMBOLIC_OPS)
    else:
        # Trace parsing failed entirely; fall back to raw extraction
        answer = extract_answer_from_model(output)
        stats = {
            "total_steps": 0,
            "symbolic_executions": 0,
            "exec_errors": 0,
            "check_failures": 0,
            "parse_errors": 1,
        }
        return {
            "predicted_answer": answer,
            "raw_outputs": all_outputs,
            "output_tokens": total_tokens,
            "repair_attempts": 0,
            "trace_steps": [],
            "stats": stats,
            "final_state": state.snapshot(),
        }

    # Repair loop
    repair_attempts = 0
    while has_errors(steps) and repair_attempts < max_repairs:
        repair_attempts += 1

        repair_prompt_text = S3MATH_REPAIR.format(
            problem=problem,
            trace_with_errors=format_trace_with_errors(steps),
            state_snapshot=state.snapshot(),
            error_summary=format_error_summary(steps),
        )

        repair_messages = [
            {"role": "system", "content": S3MATH_SYSTEM},
            {"role": "user", "content": repair_prompt_text},
        ]
        repair_prompt = model.build_prompt(repair_messages, enable_thinking=False)
        repair_output = model.generate(
            [repair_prompt], **{k: v for k, v in cfg.items()}
        )[0]

        total_tokens += model.count_tokens(repair_output)
        all_outputs.append(repair_output)

        # Re-parse and re-execute with fresh state
        steps = parse_trace(repair_output)
        state = SymbolicState()
        if steps:
            answer, steps, stats = execute_trace(steps, state, SYMBOLIC_OPS)
        else:
            answer = extract_answer_from_model(repair_output)

    # Final answer extraction
    if answer is None:
        answer = extract_answer_from_trace(steps, state)
    if answer is None:
        # Last fallback: extract from raw output
        answer = extract_answer_from_model(all_outputs[-1])

    step_summaries = [
        {
            "op": s.op_type.value,
            "raw": s.raw_line,
            "result": s.symbolic_result,
            "executed": s.executed,
            "symbolic": s.routed_symbolic,
            "error": s.error,
        }
        for s in steps
    ]

    return {
        "predicted_answer": answer,
        "raw_outputs": all_outputs,
        "output_tokens": total_tokens,
        "repair_attempts": repair_attempts,
        "trace_steps": step_summaries,
        "stats": stats,
        "final_state": state.snapshot(),
    }


def evaluate(model, data, batch_size=1):
    """Run S3-Math evaluation.

    Note: batch_size is effectively 1 because the repair loop requires
    per-problem feedback. The initial generation could be batched, but
    keeping it simple for correctness.
    """
    results = []

    for ex in tqdm(data, desc="S3-Math solving"):
        result = solve_one(model, ex["problem"])

        correct = compare_answers(result["predicted_answer"], ex["gold"], ex["source"])

        results.append(
            {
                "idx": ex.get("idx"),
                "problem": ex["problem"],
                "gold_answer": ex["gold"],
                "predicted_answer": result["predicted_answer"],
                "correct": correct,
                "raw_outputs": result["raw_outputs"],
                "output_tokens": result["output_tokens"],
                "repair_attempts": result["repair_attempts"],
                "trace_steps": result["trace_steps"],
                "stats": result["stats"],
                "final_state": result["final_state"],
            }
        )

    return results


def compute_summary(results):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    avg_tokens = sum(r["output_tokens"] for r in results) / max(total, 1)

    # S3-Math specific metrics
    total_sym_exec = sum(r["stats"]["symbolic_executions"] for r in results)
    total_exec_errors = sum(r["stats"]["exec_errors"] for r in results)
    total_check_failures = sum(r["stats"]["check_failures"] for r in results)
    total_parse_errors = sum(r["stats"]["parse_errors"] for r in results)

    needed_repair = [r for r in results if r["repair_attempts"] > 0]
    repair_success = sum(1 for r in needed_repair if r["correct"])

    problems_with_errors = sum(
        1
        for r in results
        if r["stats"]["exec_errors"] > 0
        or r["stats"]["check_failures"] > 0
        or r["stats"]["parse_errors"] > 0
    )

    return {
        "accuracy": correct / max(total, 1),
        "total": total,
        "correct": correct,
        "avg_tokens": round(avg_tokens, 1),
        "avg_symbolic_calls": round(total_sym_exec / max(total, 1), 1),
        "invalid_reasoning_rate": round(problems_with_errors / max(total, 1), 4),
        "repair_attempts_total": sum(r["repair_attempts"] for r in results),
        "repair_needed": len(needed_repair),
        "repair_success_rate": round(repair_success / max(len(needed_repair), 1), 4),
        "exec_errors_total": total_exec_errors,
        "check_failures_total": total_check_failures,
        "parse_errors_total": total_parse_errors,
    }


def main():
    parser = argparse.ArgumentParser(description="S3-Math evaluation")
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
    parser.add_argument("--max_repairs", type=int, default=S3MATH_MAX_REPAIRS)
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

        results = evaluate(model, data)
        summary = compute_summary(results)

        print(
            f"Accuracy: {summary['accuracy']:.4f} ({summary['correct']}/{summary['total']})"
        )
        print(f"Avg tokens: {summary['avg_tokens']}")
        print(f"Avg symbolic calls: {summary['avg_symbolic_calls']}")
        print(f"Invalid reasoning rate: {summary['invalid_reasoning_rate']:.4f}")
        print(
            f"Repair success rate: {summary['repair_success_rate']:.4f} "
            f"({summary['repair_needed']} needed repair)"
        )

        output = {
            "method": METHOD,
            "model": MODELS[args.model],
            "dataset": ds_name,
            "timestamp": datetime.now().isoformat(),
            "config": {"max_repairs": args.max_repairs},
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
