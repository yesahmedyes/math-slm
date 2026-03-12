"""S3-Math ablation studies (Section 7 of the paper).

Four ablations, each removing one component:
1. no_selective  - Route ALL ops to SymPy (vs selective routing)
2. no_symbolic   - No SymPy execution; trust model arithmetic
3. no_typing     - Free-form CoT prompt instead of typed traces
4. no_repair     - No repair loop on errors

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_ablations.py --model 4B --dataset gsm8k --ablation no_selective
    CUDA_VISIBLE_DEVICES=1 python run_ablations.py --model 4B --dataset gsm8k --ablation all
"""

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm

from config import MODELS, SAMPLING_CONFIGS, S3MATH_MAX_REPAIRS
from inference import load_model
from data_loader import load_dataset_by_name
from prompts import (
    S3MATH_SYSTEM,
    S3MATH_USER,
    S3MATH_REPAIR,
    ABLATION_NO_TYPING_SYSTEM,
    ABLATION_NO_TYPING_USER,
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
    ALL_OPS,
)


@dataclass
class AblationConfig:
    name: str
    use_typed_traces: bool = True
    use_symbolic_exec: bool = True
    use_selective_routing: bool = True
    use_repair_loop: bool = True


ABLATION_CONFIGS = {
    "full": AblationConfig(name="s3math_full"),
    "no_selective": AblationConfig(name="no_selective", use_selective_routing=False),
    "no_symbolic": AblationConfig(name="no_symbolic", use_symbolic_exec=False),
    "no_typing": AblationConfig(name="no_typing", use_typed_traces=False),
    "no_repair": AblationConfig(name="no_repair", use_repair_loop=False),
}


def solve_one(model, problem: str, ablation: AblationConfig) -> dict:
    """Solve a single problem with the given ablation configuration."""
    cfg = SAMPLING_CONFIGS["s3math"]
    max_repairs = S3MATH_MAX_REPAIRS if ablation.use_repair_loop else 0

    # Choose prompt based on typing ablation
    if ablation.use_typed_traces:
        messages = build_messages(S3MATH_SYSTEM, S3MATH_USER, problem)
    else:
        messages = build_messages(
            ABLATION_NO_TYPING_SYSTEM, ABLATION_NO_TYPING_USER, problem
        )

    prompt = model.build_prompt(messages, enable_thinking=False)
    output = model.generate([prompt], **{k: v for k, v in cfg.items()})[0]

    total_tokens = model.count_tokens(output)
    all_outputs = [output]

    # Choose routing
    if not ablation.use_selective_routing:
        symbolic_ops = ALL_OPS  # Route everything to SymPy
    elif not ablation.use_symbolic_exec:
        symbolic_ops = set()  # Route nothing to SymPy
    else:
        symbolic_ops = SYMBOLIC_OPS

    # Parse trace
    if ablation.use_typed_traces:
        steps = parse_trace(output)
    else:
        # Best-effort: try to parse any structured content
        steps = parse_trace(output)

    state = SymbolicState()

    if steps and ablation.use_symbolic_exec:
        answer, steps, stats = execute_trace(steps, state, symbolic_ops)
    elif steps:
        # No symbolic exec: extract answer from trace without executing
        answer = extract_answer_from_trace(steps, state)
        stats = {
            "total_steps": len(steps),
            "symbolic_executions": 0,
            "exec_errors": 0,
            "check_failures": 0,
            "parse_errors": 0,
        }
    else:
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
            "stats": stats,
        }

    # Repair loop (if enabled)
    repair_attempts = 0
    while (
        ablation.use_repair_loop and has_errors(steps) and repair_attempts < max_repairs
    ):
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

        steps = parse_trace(repair_output)
        state = SymbolicState()
        if steps and ablation.use_symbolic_exec:
            answer, steps, stats = execute_trace(steps, state, symbolic_ops)
        elif steps:
            answer = extract_answer_from_trace(steps, state)
        else:
            answer = extract_answer_from_model(repair_output)

    if answer is None:
        answer = extract_answer_from_model(all_outputs[-1])

    return {
        "predicted_answer": answer,
        "raw_outputs": all_outputs,
        "output_tokens": total_tokens,
        "repair_attempts": repair_attempts,
        "stats": stats,
    }


def evaluate(model, data, ablation: AblationConfig):
    results = []
    for ex in tqdm(data, desc=f"Ablation: {ablation.name}"):
        result = solve_one(model, ex["problem"], ablation)
        correct = compare_answers(result["predicted_answer"], ex["gold"], ex["source"])
        results.append(
            {
                "idx": ex.get("idx"),
                "problem": ex["problem"],
                "gold_answer": ex["gold"],
                "predicted_answer": result["predicted_answer"],
                "correct": correct,
                "output_tokens": result["output_tokens"],
                "repair_attempts": result["repair_attempts"],
                "stats": result["stats"],
            }
        )
    return results


def compute_summary(results):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    avg_tokens = sum(r["output_tokens"] for r in results) / max(total, 1)
    avg_sym = sum(r["stats"]["symbolic_executions"] for r in results) / max(total, 1)

    needed_repair = [r for r in results if r["repair_attempts"] > 0]
    repair_success = sum(1 for r in needed_repair if r["correct"])

    return {
        "accuracy": correct / max(total, 1),
        "total": total,
        "correct": correct,
        "avg_tokens": round(avg_tokens, 1),
        "avg_symbolic_calls": round(avg_sym, 1),
        "repair_success_rate": round(repair_success / max(len(needed_repair), 1), 4)
        if needed_repair
        else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="S3-Math ablation studies")
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
    parser.add_argument(
        "--ablation", required=True, choices=list(ABLATION_CONFIGS.keys()) + ["all"]
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    model = load_model(args.model)

    if args.dataset == "all":
        datasets = load_dataset_by_name("all", args.max_samples)
    else:
        datasets = {args.dataset: load_dataset_by_name(args.dataset, args.max_samples)}

    ablations = (
        list(ABLATION_CONFIGS.values())
        if args.ablation == "all"
        else [ABLATION_CONFIGS[args.ablation]]
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for ablation in ablations:
        for ds_name, data in datasets.items():
            print(f"\n{'=' * 60}")
            print(f"Ablation: {ablation.name} | model={args.model} | dataset={ds_name}")
            print(
                f"  typed_traces={ablation.use_typed_traces}, "
                f"symbolic_exec={ablation.use_symbolic_exec}, "
                f"selective={ablation.use_selective_routing}, "
                f"repair={ablation.use_repair_loop}"
            )
            print(f"{'=' * 60}")

            results = evaluate(model, data, ablation)
            summary = compute_summary(results)

            print(
                f"Accuracy: {summary['accuracy']:.4f} "
                f"({summary['correct']}/{summary['total']})"
            )

            output = {
                "method": f"ablation_{ablation.name}",
                "model": MODELS[args.model],
                "dataset": ds_name,
                "timestamp": datetime.now().isoformat(),
                "ablation_config": {
                    "use_typed_traces": ablation.use_typed_traces,
                    "use_symbolic_exec": ablation.use_symbolic_exec,
                    "use_selective_routing": ablation.use_selective_routing,
                    "use_repair_loop": ablation.use_repair_loop,
                },
                "summary": summary,
                "samples": results,
            }

            fname = f"ablation_{ablation.name}_{args.model}_{ds_name}.json"
            path = os.path.join(args.output_dir, fname)
            with open(path, "w") as f:
                json.dump(output, f, indent=2)
            print(f"Results saved to {path}")


if __name__ == "__main__":
    main()
