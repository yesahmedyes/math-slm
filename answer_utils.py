"""Answer extraction and comparison utilities for GSM8K and MATH benchmarks."""

import re
import sympy
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)


def extract_gsm8k_gold(answer_str: str) -> str:
    """Extract gold answer from GSM8K '#### number' format."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_str)
    if match:
        return match.group(1).replace(",", "")
    return None


def extract_math_gold(solution_str: str) -> str:
    """Extract gold answer from MATH \\boxed{} format, handling nested braces."""
    # Find the last \boxed{...} in the solution
    idx = solution_str.rfind("\\boxed{")
    if idx == -1:
        return None
    # Match balanced braces
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(solution_str)):
        if solution_str[i] == "{":
            depth += 1
        elif solution_str[i] == "}":
            if depth == 0:
                return solution_str[start:i].strip()
            depth -= 1
    return None


def extract_answer_from_model(text: str) -> str:
    """Extract numerical answer from model output (works for any method)."""
    if not text:
        return None

    # Priority 1: ANSWER: line (S3-Math format)
    m = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", text)
    if m:
        val = m.group(1).strip()
        # Clean up any trailing text
        val = re.sub(r"\s*#.*$", "", val)
        return val

    # Priority 2: \boxed{...} (MATH format)
    idx = text.rfind("\\boxed{")
    if idx != -1:
        depth = 0
        start = idx + len("\\boxed{")
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                if depth == 0:
                    return text[start:i].strip()
                depth -= 1

    # Priority 3: "the answer is X" patterns
    m = re.search(
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*\$?\\?(?:boxed\{)?(-?[\d,./]+\.?\d*)\}?\$?",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).replace(",", "")

    # Priority 4: Last number in text (common for direct answer)
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def normalize_answer(ans: str) -> str:
    """Normalize an answer string for comparison."""
    if ans is None:
        return None
    ans = ans.strip()
    # Remove trailing period
    ans = ans.rstrip(".")
    # Remove dollar signs and LaTeX wrappers
    ans = ans.replace("$", "").replace("\\", "")
    # Remove commas in numbers
    ans = ans.replace(",", "")
    # Normalize whitespace
    ans = re.sub(r"\s+", " ", ans).strip()
    return ans


def _try_parse_number(s: str):
    """Try to parse a string as a number (int, float, or fraction)."""
    s = s.strip()
    # Handle fractions like "3/4"
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                pass
    try:
        return float(s)
    except ValueError:
        return None


def compare_answers(predicted: str, gold: str, source: str = "gsm8k") -> bool:
    """Compare predicted and gold answers. Returns True if they match."""
    if predicted is None or gold is None:
        return False

    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    if pred_norm is None or gold_norm is None:
        return False

    # Direct string match
    if pred_norm == gold_norm:
        return True

    # Numeric comparison
    pred_num = _try_parse_number(pred_norm)
    gold_num = _try_parse_number(gold_norm)

    if pred_num is not None and gold_num is not None:
        # Allow small floating point tolerance
        if abs(gold_num) < 1e-9:
            return abs(pred_num - gold_num) < 1e-6
        return abs(pred_num - gold_num) / max(abs(gold_num), 1e-9) < 1e-4

    # For MATH: try symbolic equivalence via SymPy
    if source.startswith("math"):
        try:
            transformations = standard_transformations + (
                implicit_multiplication_application,
            )
            pred_expr = parse_expr(pred_norm, transformations=transformations)
            gold_expr = parse_expr(gold_norm, transformations=transformations)
            diff = sympy.simplify(pred_expr - gold_expr)
            return diff == 0
        except Exception:
            pass

    return False
