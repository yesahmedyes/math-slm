"""S3-Math symbolic engine: trace parser, state tracking, SymPy executor, and repair loop."""

import re
import signal
from enum import Enum
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Optional

import sympy
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

from config import S3MATH_SYMPY_TIMEOUT

# =============================================================================
# Operation types
# =============================================================================


class OpType(Enum):
    DEFINE = "DEFINE"
    ASSUME = "ASSUME"
    EQUATION = "EQUATION"
    SIMPLIFY = "SIMPLIFY"
    SUBSTITUTE = "SUBSTITUTE"
    CHECK = "CHECK"
    SOLVE = "SOLVE"
    ANSWER = "ANSWER"


# Which operations get routed to SymPy (selective formalization)
SYMBOLIC_OPS = {OpType.SOLVE, OpType.SIMPLIFY, OpType.SUBSTITUTE, OpType.CHECK}
NEURAL_OPS = {OpType.DEFINE, OpType.ASSUME, OpType.EQUATION, OpType.ANSWER}
ALL_OPS = set(OpType)


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class TraceStep:
    op_type: OpType
    raw_line: str
    lhs: str = ""
    rhs: str = ""
    symbolic_result: Optional[str] = None
    executed: bool = False
    routed_symbolic: bool = False
    error: Optional[str] = None


@dataclass
class SymbolicState:
    variables: dict = field(default_factory=dict)  # name -> SymPy expr or number
    equations: list = field(default_factory=list)  # list of SymPy Eq objects
    constraints: list = field(default_factory=list)  # list of SymPy expressions

    def snapshot(self) -> str:
        """Human-readable snapshot of current state."""
        lines = []
        for k, v in self.variables.items():
            lines.append(f"  {k} = {v}")
        if self.equations:
            lines.append("  Equations:")
            for eq in self.equations:
                lines.append(f"    {eq}")
        if self.constraints:
            lines.append("  Constraints:")
            for c in self.constraints:
                lines.append(f"    {c}")
        return "\n".join(lines) if lines else "  (empty)"


# =============================================================================
# Timeout utility
# =============================================================================


@contextmanager
def timeout(seconds):
    """Timeout context manager using SIGALRM (Unix only)."""

    def handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# =============================================================================
# Safe expression parsing
# =============================================================================

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def safe_parse(expr_str: str, local_vars: dict) -> sympy.Expr:
    """Parse a string into a SymPy expression using the current variable context."""
    expr_str = expr_str.strip()

    # Build local dict: known variable names -> their SymPy values
    local_dict = {}
    for k, v in local_vars.items():
        if isinstance(v, sympy.Basic):
            local_dict[k] = v
        else:
            try:
                local_dict[k] = sympy.Rational(str(v))
            except (ValueError, TypeError):
                local_dict[k] = sympy.Symbol(k)

    try:
        return parse_expr(
            expr_str, local_dict=local_dict, transformations=TRANSFORMATIONS
        )
    except Exception:
        pass

    # Fallback: try as a simple number
    try:
        if "/" in expr_str:
            parts = expr_str.split("/")
            return sympy.Rational(parts[0].strip(), parts[1].strip())
        return sympy.Rational(expr_str)
    except (ValueError, TypeError, IndexError):
        pass

    # Last resort: treat as a symbol
    return sympy.Symbol(expr_str.replace(" ", "_"))


# =============================================================================
# Trace parsing
# =============================================================================

TRACE_PATTERNS = {
    OpType.DEFINE: re.compile(r"DEFINE:\s*(\w+)\s*=\s*(.+)", re.IGNORECASE),
    OpType.ASSUME: re.compile(r"ASSUME:\s*(.+)", re.IGNORECASE),
    OpType.EQUATION: re.compile(r"EQUATION:\s*(\w+)\s*=\s*(.+)", re.IGNORECASE),
    OpType.SIMPLIFY: re.compile(r"SIMPLIFY:\s*(.+?)\s*(?:->|=)\s*(.+)", re.IGNORECASE),
    OpType.SUBSTITUTE: re.compile(
        r"SUBSTITUTE:\s*(.+?)\s*(?:->|=)\s*(.+)", re.IGNORECASE
    ),
    OpType.CHECK: re.compile(r"CHECK:\s*(.+?)\s*->\s*(.+)", re.IGNORECASE),
    OpType.SOLVE: re.compile(r"SOLVE:\s*(.+?)\s*->\s*(.+)", re.IGNORECASE),
    OpType.ANSWER: re.compile(r"ANSWER:\s*(.+)", re.IGNORECASE),
}

# Alternative patterns (model may use slightly different formats)
ALT_PATTERNS = {
    OpType.DEFINE: re.compile(r"DEFINE\((\w+),\s*(.+?)\)", re.IGNORECASE),
    OpType.EQUATION: re.compile(r"EQUATION\((.+?),\s*(.+?)\)", re.IGNORECASE),
    OpType.SIMPLIFY: re.compile(r"SIMPLIFY\((.+?)\)\s*(?:->|=)\s*(.+)", re.IGNORECASE),
    OpType.SOLVE: re.compile(r"SOLVE\((.+?)\)\s*(?:->|=)\s*(.+)", re.IGNORECASE),
    OpType.ANSWER: re.compile(r"ANSWER\((.+?)\)", re.IGNORECASE),
}


def parse_trace(model_output: str) -> list[TraceStep]:
    """Parse typed operation trace from model output."""
    steps = []
    for line in model_output.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        matched = False
        # Try primary patterns first
        for op_type, pattern in TRACE_PATTERNS.items():
            m = pattern.match(line)
            if m:
                groups = m.groups()
                lhs = groups[0].strip() if len(groups) > 0 else ""
                rhs = groups[1].strip() if len(groups) > 1 else ""
                steps.append(
                    TraceStep(op_type=op_type, raw_line=line, lhs=lhs, rhs=rhs)
                )
                matched = True
                break

        # Try alternative patterns
        if not matched:
            for op_type, pattern in ALT_PATTERNS.items():
                m = pattern.match(line)
                if m:
                    groups = m.groups()
                    lhs = groups[0].strip() if len(groups) > 0 else ""
                    rhs = groups[1].strip() if len(groups) > 1 else ""
                    steps.append(
                        TraceStep(op_type=op_type, raw_line=line, lhs=lhs, rhs=rhs)
                    )
                    matched = True
                    break

        # Skip non-matching lines (free-form text the model added)

    return steps


# =============================================================================
# Trace execution
# =============================================================================


def execute_trace(
    steps: list[TraceStep], state: SymbolicState, symbolic_ops: set = None
) -> tuple[str, list[TraceStep], dict]:
    """Execute all trace steps, updating the symbolic state.

    Args:
        steps: parsed trace steps
        state: symbolic state to update
        symbolic_ops: which op types to route to SymPy (default: SYMBOLIC_OPS)

    Returns:
        (final_answer, executed_steps, stats)
    """
    if symbolic_ops is None:
        symbolic_ops = SYMBOLIC_OPS

    final_answer = None
    stats = {
        "total_steps": len(steps),
        "symbolic_executions": 0,
        "exec_errors": 0,
        "check_failures": 0,
        "parse_errors": 0,
    }

    for step in steps:
        try:
            route_symbolic = step.op_type in symbolic_ops

            if step.op_type == OpType.DEFINE:
                if route_symbolic:
                    with timeout(S3MATH_SYMPY_TIMEOUT):
                        expr = safe_parse(step.rhs, state.variables)
                        # Try to evaluate numerically
                        evaled = expr.subs(
                            {
                                sympy.Symbol(k): v
                                for k, v in state.variables.items()
                                if isinstance(
                                    v, (int, float, sympy.Number, sympy.Rational)
                                )
                            }
                        )
                        state.variables[step.lhs] = evaled
                        step.symbolic_result = str(evaled)
                    stats["symbolic_executions"] += 1
                else:
                    # Neural: just store the raw expression, try simple eval
                    try:
                        val = _try_eval_simple(step.rhs, state.variables)
                        state.variables[step.lhs] = val
                    except Exception:
                        state.variables[step.lhs] = step.rhs
                step.executed = True
                step.routed_symbolic = route_symbolic

            elif step.op_type == OpType.EQUATION:
                if route_symbolic:
                    with timeout(S3MATH_SYMPY_TIMEOUT):
                        rhs_expr = safe_parse(step.rhs, state.variables)
                        evaled = rhs_expr.subs(
                            {
                                sympy.Symbol(k): v
                                for k, v in state.variables.items()
                                if isinstance(
                                    v, (int, float, sympy.Number, sympy.Rational)
                                )
                            }
                        )
                        state.variables[step.lhs] = evaled
                        lhs_sym = sympy.Symbol(step.lhs)
                        state.equations.append(sympy.Eq(lhs_sym, evaled))
                        step.symbolic_result = str(evaled)
                    stats["symbolic_executions"] += 1
                else:
                    try:
                        val = _try_eval_simple(step.rhs, state.variables)
                        state.variables[step.lhs] = val
                    except Exception:
                        state.variables[step.lhs] = step.rhs
                step.executed = True
                step.routed_symbolic = route_symbolic

            elif step.op_type == OpType.SIMPLIFY:
                if route_symbolic:
                    with timeout(S3MATH_SYMPY_TIMEOUT):
                        expr = safe_parse(step.lhs, state.variables)
                        result = sympy.simplify(
                            expr.subs(
                                {
                                    sympy.Symbol(k): v
                                    for k, v in state.variables.items()
                                    if isinstance(
                                        v, (int, float, sympy.Number, sympy.Rational)
                                    )
                                }
                            )
                        )
                        step.symbolic_result = str(result)
                        # Update state if there's an assignment pattern
                        parts = step.lhs.split("=")
                        if len(parts) == 2:
                            var_name = parts[0].strip()
                            if var_name.isidentifier():
                                state.variables[var_name] = result
                    stats["symbolic_executions"] += 1
                step.executed = True
                step.routed_symbolic = route_symbolic

            elif step.op_type == OpType.SUBSTITUTE:
                if route_symbolic:
                    with timeout(S3MATH_SYMPY_TIMEOUT):
                        expr = safe_parse(step.lhs, state.variables)
                        result = expr.subs(
                            {
                                sympy.Symbol(k): v
                                for k, v in state.variables.items()
                                if isinstance(
                                    v, (int, float, sympy.Number, sympy.Rational)
                                )
                            }
                        )
                        step.symbolic_result = str(result)
                    stats["symbolic_executions"] += 1
                step.executed = True
                step.routed_symbolic = route_symbolic

            elif step.op_type == OpType.SOLVE:
                if route_symbolic:
                    with timeout(S3MATH_SYMPY_TIMEOUT):
                        var_name = step.lhs.strip()
                        sym = sympy.Symbol(var_name)
                        # Try to solve from accumulated equations
                        if state.equations:
                            solutions = sympy.solve(state.equations, sym)
                            if isinstance(solutions, dict) and sym in solutions:
                                result = solutions[sym]
                            elif isinstance(solutions, list) and solutions:
                                result = solutions[0]
                            else:
                                result = None
                        else:
                            result = None

                        if result is not None:
                            state.variables[var_name] = result
                            step.symbolic_result = str(result)
                        else:
                            # Fallback: use the model's claimed answer
                            step.symbolic_result = step.rhs
                            try:
                                state.variables[var_name] = sympy.Rational(step.rhs)
                            except (ValueError, TypeError):
                                state.variables[var_name] = step.rhs
                    stats["symbolic_executions"] += 1
                else:
                    # Neural: trust model's answer
                    try:
                        state.variables[step.lhs.strip()] = float(step.rhs)
                    except (ValueError, TypeError):
                        state.variables[step.lhs.strip()] = step.rhs
                step.executed = True
                step.routed_symbolic = route_symbolic

            elif step.op_type == OpType.CHECK:
                if route_symbolic:
                    with timeout(S3MATH_SYMPY_TIMEOUT):
                        # Try to evaluate the condition
                        cond_str = step.lhs
                        # Replace variable names with values
                        for k, v in state.variables.items():
                            cond_str = re.sub(
                                r"\b" + re.escape(k) + r"\b", str(v), cond_str
                            )
                        try:
                            check_result = bool(
                                eval(
                                    cond_str,
                                    {"__builtins__": {}},
                                    {"abs": abs, "min": min, "max": max},
                                )
                            )
                        except Exception:
                            check_result = None
                        step.symbolic_result = str(check_result)
                        if check_result is False:
                            step.error = "Check failed"
                            stats["check_failures"] += 1
                    stats["symbolic_executions"] += 1
                step.executed = True
                step.routed_symbolic = route_symbolic

            elif step.op_type == OpType.ASSUME:
                state.constraints.append(step.lhs)
                step.executed = True

            elif step.op_type == OpType.ANSWER:
                # Try to resolve the answer from state
                answer_str = step.lhs.strip()
                if answer_str in state.variables:
                    final_answer = str(state.variables[answer_str])
                else:
                    final_answer = answer_str
                step.executed = True

        except TimeoutError as e:
            step.error = str(e)
            stats["exec_errors"] += 1
        except Exception as e:
            step.error = str(e)
            stats["exec_errors"] += 1

    return final_answer, steps, stats


def _try_eval_simple(expr_str: str, variables: dict):
    """Try simple arithmetic evaluation using known variable values."""
    local = {}
    for k, v in variables.items():
        if isinstance(v, (int, float)):
            local[k] = v
        elif isinstance(v, sympy.Basic):
            try:
                local[k] = float(v)
            except (TypeError, ValueError):
                local[k] = v
        else:
            try:
                local[k] = float(str(v))
            except (ValueError, TypeError):
                pass
    return eval(expr_str, {"__builtins__": {}}, local)


# =============================================================================
# Repair utilities
# =============================================================================


def has_errors(steps: list[TraceStep]) -> bool:
    """Check if any step has errors."""
    return any(s.error is not None for s in steps)


def format_error_summary(steps: list[TraceStep]) -> str:
    """Format a summary of errors in the trace."""
    errors = []
    for i, step in enumerate(steps):
        if step.error:
            errors.append(f"Step {i + 1} ({step.op_type.value}): {step.error}")
    return "\n".join(errors) if errors else "No errors"


def format_trace_with_errors(steps: list[TraceStep]) -> str:
    """Format trace showing step status."""
    lines = []
    for i, step in enumerate(steps):
        status = "OK" if step.executed and not step.error else f"ERROR: {step.error}"
        result_str = f" = {step.symbolic_result}" if step.symbolic_result else ""
        lines.append(f"  Step {i + 1}: {step.raw_line}  [{status}]{result_str}")
    return "\n".join(lines)


# =============================================================================
# Extract answer from raw output (fallback when trace parsing fails)
# =============================================================================


def extract_answer_from_trace(steps: list[TraceStep], state: SymbolicState) -> str:
    """Try to get the answer from parsed trace, falling back to state."""
    # Check ANSWER steps
    for step in steps:
        if step.op_type == OpType.ANSWER:
            val = step.lhs.strip()
            if val in state.variables:
                return str(state.variables[val])
            return val

    # Check SOLVE steps (last one)
    for step in reversed(steps):
        if step.op_type == OpType.SOLVE and step.symbolic_result:
            return step.symbolic_result

    return None
