"""Method 8: ToRA (Tool-integrated Reasoning Agent) evaluation.

Multi-round interleaved reasoning framework adapted from Gou et al. (2024)
"ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving":
  - The model alternates between natural language reasoning (Thought) and
    Python code execution (Program/Output) in multiple rounds.
  - After each code block, the execution output is fed back to the model
    so it can refine its reasoning or produce the final answer.
  - Unlike single-shot PAL/PoT, ToRA can observe intermediate results and
    adjust its approach across multiple rounds.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_tora.py --model 4B --dataset gsm8k
"""

import argparse
import json
import os
import re
from datetime import datetime
from tqdm import tqdm

from config import MODELS, SAMPLING_CONFIGS
from inference import load_model
from data_loader import load_dataset_by_name
from prompts import TORA_SYSTEM, TORA_USER, TORA_CONTINUE_USER
from answer_utils import extract_answer_from_model, compare_answers
from sandbox import execute_code

METHOD = "tora"


def _has_code_block(text: str) -> bool:
    """Check if text contains a ```python code block."""
    return bool(re.search(r"```python\s*\n", text))


def _extract_all_code_blocks(text: str) -> list[str]:
    """Extract all ```python ... ``` blocks from text."""
    return re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)


def _extract_last_code_block(text: str) -> str | None:
    """Extract the last ```python ... ``` block from text."""
    blocks = _extract_all_code_blocks(text)
    return blocks[-1].strip() if blocks else None


def _format_output_block(exec_result: dict) -> str:
    """Format execution result as an ```output``` block for feeding back."""
    if exec_result["success"] and exec_result["stdout"]:
        content = exec_result["stdout"]
    elif exec_result["error"]:
        content = exec_result["error"]
    elif exec_result["stderr"]:
        content = exec_result["stderr"]
    else:
        content = "(no output)"
    return f"```output\n{content}\n```"


def solve_with_tora(model, problem: str, cfg: dict) -> dict:
    """Solve a single problem using ToRA's multi-round interleaved format.

    Returns a dict with all intermediate state and the final answer.
    """
    max_rounds = cfg.get("max_rounds", 3)

    # Build initial prompt
    messages = [
        {"role": "system", "content": TORA_SYSTEM},
        {"role": "user", "content": TORA_USER.format(problem=problem)},
    ]

    all_outputs = []
    all_codes = []
    all_exec_results = []
    reasoning_rounds = 0
    total_text = ""

    for round_idx in range(max_rounds):
        # Generate model response
        prompt = model.build_prompt(messages, enable_thinking=False)
        outputs = model.generate(
            [prompt],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            max_tokens=cfg["max_tokens"],
        )
        output = outputs[0]
        all_outputs.append(output)
        total_text += output

        # Check if the output contains a code block
        code = _extract_last_code_block(output)

        if code is None:
            # No code block — model is done reasoning, extract final answer
            break

        # Execute the code
        exec_result = execute_code(code)
        all_codes.append(code)
        all_exec_results.append(exec_result)
        reasoning_rounds += 1

        # Format the output block and feed back to the model
        output_block = _format_output_block(exec_result)

        # Append the assistant's response and the output as a continuation
        messages.append({"role": "assistant", "content": output})
        messages.append(
            {
                "role": "user",
                "content": TORA_CONTINUE_USER.format(output_block=output_block),
            }
        )

    # Extract the final answer from the full accumulated text
    predicted = extract_answer_from_model(total_text)

    # If no answer found in NL text, try extracting from last successful execution
    if predicted is None and all_exec_results:
        for er in reversed(all_exec_results):
            if er["success"] and er["stdout"]:
                predicted = extract_answer_from_model(er["stdout"])
                if predicted is not None:
                    break

    return {
        "predicted": predicted,
        "all_outputs": all_outputs,
        "all_codes": all_codes,
        "all_exec_results": all_exec_results,
        "reasoning_rounds": reasoning_rounds,
        "total_text": total_text,
    }


def evaluate(model, data, batch_size=1):
    """Evaluate ToRA on a dataset.

    Note: batch_size is effectively 1 because ToRA's multi-round loop is
    inherently sequential per sample (each round depends on prior output).
    """
    cfg = SAMPLING_CONFIGS[METHOD]
    results = []

    for ex in tqdm(data, desc="ToRA"):
        solve_result = solve_with_tora(model, ex["problem"], cfg)

        correct = compare_answers(solve_result["predicted"], ex["gold"], ex["source"])

        total_tokens = sum(model.count_tokens(o) for o in solve_result["all_outputs"])

        exec_successes = [er["success"] for er in solve_result["all_exec_results"]]

        results.append(
            {
                "idx": ex.get("idx"),
                "problem": ex["problem"],
                "gold_answer": ex["gold"],
                "predicted_answer": solve_result["predicted"],
                "correct": correct,
                "raw_output": solve_result["all_outputs"][0]
                if solve_result["all_outputs"]
                else "",
                "full_trajectory": solve_result["total_text"],
                "codes": solve_result["all_codes"],
                "exec_results": [
                    {
                        "stdout": er["stdout"],
                        "stderr": er["stderr"],
                        "success": er["success"],
                    }
                    for er in solve_result["all_exec_results"]
                ],
                "reasoning_rounds": solve_result["reasoning_rounds"],
                "all_exec_success": all(exec_successes) if exec_successes else False,
                "any_exec_success": any(exec_successes) if exec_successes else False,
                "output_tokens": total_tokens,
            }
        )

    return results


def compute_summary(results):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    avg_tokens = sum(r["output_tokens"] for r in results) / max(total, 1)
    avg_rounds = sum(r["reasoning_rounds"] for r in results) / max(total, 1)
    any_exec = sum(1 for r in results if r["any_exec_success"])
    all_exec = sum(1 for r in results if r["all_exec_success"])
    multi_round = sum(1 for r in results if r["reasoning_rounds"] > 1)

    # Total code executions across all samples
    total_code_calls = sum(r["reasoning_rounds"] for r in results)

    return {
        "accuracy": correct / max(total, 1),
        "total": total,
        "correct": correct,
        "avg_tokens": round(avg_tokens, 1),
        "avg_reasoning_rounds": round(avg_rounds, 2),
        "avg_symbolic_calls": round(total_code_calls / max(total, 1), 2),
        "samples_with_any_exec_success": any_exec,
        "samples_with_all_exec_success": all_exec,
        "multi_round_samples": multi_round,
        "exec_success_rate": round(any_exec / max(total, 1), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="ToRA evaluation")
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
    parser.add_argument("--batch_size", type=int, default=1)
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
        print(f"Avg reasoning rounds: {summary['avg_reasoning_rounds']}")
        print(
            f"Multi-round samples: {summary['multi_round_samples']}/{summary['total']}"
        )
        print(f"Exec success rate: {summary['exec_success_rate']:.4f}")
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
