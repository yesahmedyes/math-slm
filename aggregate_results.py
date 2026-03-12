"""Aggregate results from all experiments into paper tables.

Usage:
    python aggregate_results.py --results_dir results/
"""

import argparse
import json
import os
import glob


def load_all_results(results_dir):
    """Load all JSON result files from the results directory."""
    results = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            data = json.load(f)
        results.append(data)
    return results


def get_accuracy(results, method, model, dataset):
    """Look up accuracy for a specific method/model/dataset combo."""
    for r in results:
        if r["method"] == method and r["model"] == model and r["dataset"] == dataset:
            return r["summary"]["accuracy"]
    return None


def print_table1(results):
    """Table 1: Main results across model sizes."""
    from config import MODELS

    methods = ["direct", "cot", "sc_cot", "code", "formal", "s3math"]
    method_names = {
        "direct": "Direct Answer",
        "cot": "CoT",
        "sc_cot": "Self-Consistent CoT",
        "code": "Code-only",
        "formal": "Full Formalization",
        "s3math": "S3-Math (ours)",
    }
    datasets = ["gsm8k", "math_algebra", "math_number_theory", "math_counting_prob"]
    model_keys = ["0.8B", "2B", "4B"]

    print("\n" + "=" * 80)
    print("TABLE 1: Overall Accuracy by Model Size")
    print("=" * 80)
    print(f"{'Method':<25} {'0.8B':>8} {'2B':>8} {'4B':>8} {'Avg':>8}")
    print("-" * 57)

    for method in methods:
        row = [method_names.get(method, method)]
        accs = []
        for mk in model_keys:
            model_name = MODELS[mk]
            # Average across all datasets
            ds_accs = []
            for ds in datasets:
                acc = get_accuracy(results, method, model_name, ds)
                if acc is not None:
                    ds_accs.append(acc)
            if ds_accs:
                avg = sum(ds_accs) / len(ds_accs)
                accs.append(avg)
                row.append(f"{avg * 100:.1f}")
            else:
                accs.append(None)
                row.append("  -")

        valid = [a for a in accs if a is not None]
        overall_avg = sum(valid) / len(valid) if valid else None
        row.append(f"{overall_avg * 100:.1f}" if overall_avg else "  -")

        print(f"{row[0]:<25} {row[1]:>8} {row[2]:>8} {row[3]:>8} {row[4]:>8}")

    print()


def print_table2(results):
    """Table 2: Accuracy by task type (largest model)."""
    from config import MODELS

    model_name = MODELS["4B"]
    methods_to_show = ["cot", "formal", "s3math"]
    method_names = {"cot": "CoT", "formal": "Full Form.", "s3math": "S3-Math"}

    task_map = {
        "Basic arithmetic": "gsm8k",
        "Algebra word problems": "math_algebra",
        "Constraint-heavy": "math_number_theory",
    }

    print("\n" + "=" * 80)
    print("TABLE 2: Accuracy by Task Type (4B model)")
    print("=" * 80)
    header = f"{'Task Type':<25}"
    for m in methods_to_show:
        header += f" {method_names[m]:>12}"
    header += f" {'Gain':>8}"
    print(header)
    print("-" * 70)

    for task_name, ds_name in task_map.items():
        row = [task_name]
        accs = {}
        for m in methods_to_show:
            acc = get_accuracy(results, m, model_name, ds_name)
            accs[m] = acc
            row.append(f"{acc * 100:.1f}" if acc else "  -")

        # Gain = S3-Math - CoT
        if accs.get("s3math") and accs.get("cot"):
            gain = (accs["s3math"] - accs["cot"]) * 100
            row.append(f"+{gain:.1f}")
        else:
            row.append("  -")

        line = f"{row[0]:<25}"
        for v in row[1:]:
            line += f" {v:>12}"
        print(line)

    print()


def print_table3(results):
    """Table 3: Robustness metrics."""
    from config import MODELS

    print("\n" + "=" * 80)
    print("TABLE 3: S3-Math Detailed Metrics")
    print("=" * 80)

    s3math_results = [r for r in results if r["method"] == "s3math"]
    if not s3math_results:
        print("  No S3-Math results found.")
        return

    print(
        f"{'Model':<20} {'Dataset':<20} {'Accuracy':>10} {'InvalidRate':>12} "
        f"{'RepairRate':>12} {'AvgTokens':>10}"
    )
    print("-" * 86)

    for r in s3math_results:
        s = r["summary"]
        model_short = r["model"].split("/")[-1]
        print(
            f"{model_short:<20} {r['dataset']:<20} "
            f"{s['accuracy'] * 100:>9.1f}% "
            f"{s.get('invalid_reasoning_rate', 0) * 100:>11.1f}% "
            f"{s.get('repair_success_rate', 0) * 100:>11.1f}% "
            f"{s.get('avg_tokens', 0):>10.0f}"
        )

    print()


def print_table4(results):
    """Table 4: Efficiency comparison."""
    from config import MODELS

    print("\n" + "=" * 80)
    print("TABLE 4: Efficiency Comparison (averaged across datasets)")
    print("=" * 80)

    methods = ["cot", "sc_cot", "formal", "s3math"]
    method_names = {
        "cot": "CoT",
        "sc_cot": "Self-Consistency (5x)",
        "formal": "Full Formalization",
        "s3math": "S3-Math (ours)",
    }

    print(f"{'Method':<25} {'Avg Tokens':>12} {'Sym. Calls':>12} {'Accuracy':>10}")
    print("-" * 61)

    for method in methods:
        method_results = [r for r in results if r["method"] == method]
        if not method_results:
            continue

        all_tokens = [r["summary"]["avg_tokens"] for r in method_results]
        all_accs = [r["summary"]["accuracy"] for r in method_results]
        sym_calls = [r["summary"].get("avg_symbolic_calls", 0) for r in method_results]

        avg_tok = sum(all_tokens) / len(all_tokens)
        avg_acc = sum(all_accs) / len(all_accs)
        avg_sym = sum(sym_calls) / len(sym_calls)

        print(
            f"{method_names.get(method, method):<25} "
            f"{avg_tok:>12.0f} "
            f"{avg_sym:>12.1f} "
            f"{avg_acc * 100:>9.1f}%"
        )

    print()


def print_ablation_table(results):
    """Ablation results table."""
    print("\n" + "=" * 80)
    print("TABLE 5: Ablation Study")
    print("=" * 80)

    ablation_results = [r for r in results if r["method"].startswith("ablation_")]
    if not ablation_results:
        print("  No ablation results found.")
        return

    ablation_names = {
        "ablation_s3math_full": "S3-Math (full)",
        "ablation_no_selective": "- Remove selective routing",
        "ablation_no_symbolic": "- Remove symbolic state",
        "ablation_no_typing": "- Remove typing",
        "ablation_no_repair": "- Remove local repair",
    }

    print(f"{'Variant':<35} {'Accuracy':>10} {'Avg Tokens':>12} {'RepairRate':>12}")
    print("-" * 71)

    for method_key in ablation_names:
        matching = [r for r in ablation_results if r["method"] == method_key]
        if not matching:
            continue

        accs = [r["summary"]["accuracy"] for r in matching]
        tokens = [r["summary"]["avg_tokens"] for r in matching]
        repairs = [r["summary"].get("repair_success_rate", 0) for r in matching]

        avg_acc = sum(accs) / len(accs)
        avg_tok = sum(tokens) / len(tokens)
        avg_rep = sum(repairs) / len(repairs)

        print(
            f"{ablation_names[method_key]:<35} "
            f"{avg_acc * 100:>9.1f}% "
            f"{avg_tok:>12.0f} "
            f"{avg_rep * 100:>11.1f}%"
        )

    print()


def export_csv(results, output_path):
    """Export all results as a CSV for further analysis."""
    import csv

    rows = []
    for r in results:
        rows.append(
            {
                "method": r["method"],
                "model": r["model"],
                "dataset": r["dataset"],
                "accuracy": r["summary"]["accuracy"],
                "total": r["summary"]["total"],
                "correct": r["summary"]["correct"],
                "avg_tokens": r["summary"].get("avg_tokens", ""),
                "avg_symbolic_calls": r["summary"].get("avg_symbolic_calls", ""),
                "invalid_rate": r["summary"].get("invalid_reasoning_rate", ""),
                "repair_success": r["summary"].get("repair_success_rate", ""),
                "exec_success": r["summary"].get("exec_success_rate", ""),
            }
        )

    if not rows:
        print("No results to export.")
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--export_csv", default=None, help="Path to export CSV")
    args = parser.parse_args()

    results = load_all_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}/")
        return

    print(f"Loaded {len(results)} result files from {args.results_dir}/")

    print_table1(results)
    print_table2(results)
    print_table3(results)
    print_table4(results)
    print_ablation_table(results)

    if args.export_csv:
        export_csv(results, args.export_csv)


if __name__ == "__main__":
    main()
