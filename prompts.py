"""Prompt templates for all 6 evaluation methods."""

# =============================================================================
# Method 1: Direct Answer
# =============================================================================

DIRECT_SYSTEM = "You are a math problem solver. Give only the final numerical answer."

DIRECT_USER = """Solve the following math problem. Output ONLY the final numerical answer, nothing else.

Problem: {problem}

Answer:"""

# =============================================================================
# Method 2: Unconstrained Chain-of-Thought
# =============================================================================

COT_SYSTEM = "You are a math problem solver. Show your step-by-step reasoning."

COT_USER = """Solve the following math problem step by step. Show your reasoning clearly, then give the final answer on the last line as "The answer is: <number>".

Problem: {problem}

Solution:"""

# =============================================================================
# Method 3: Self-Consistent CoT (same prompt as CoT, sampled multiple times)
# =============================================================================

SC_COT_SYSTEM = COT_SYSTEM
SC_COT_USER = COT_USER

# =============================================================================
# Method 4: Code-only
# =============================================================================

CODE_SYSTEM = "You are a math problem solver that writes Python code."

CODE_USER = """Solve the following math problem by writing a Python program. Your code must print ONLY the final numerical answer.

Problem: {problem}

```python
"""

# =============================================================================
# Method 5: Full Formalization (SymPy)
# =============================================================================

FORMAL_SYSTEM = "You are a math problem solver that uses SymPy for symbolic computation."

FORMAL_USER = """Solve the following math problem by writing a complete SymPy program. Define all variables as symbols, set up all equations, solve them symbolically, and print ONLY the final numerical answer.

Problem: {problem}

```python
from sympy import *
"""

# =============================================================================
# Method 6: S3-Math (Selective Neuro-Symbolic State Scaffolding)
# =============================================================================

S3MATH_SYSTEM = """You are a precise math reasoning assistant that solves problems using typed operation traces with a symbolic state.

RULES:
1. First identify all quantities, variables, and unknowns
2. Write your solution as a sequence of typed operations, one per line
3. Use descriptive variable names
4. End with ANSWER containing only the final numerical value

AVAILABLE OPERATIONS:
- DEFINE: variable_name = value_or_expression
- ASSUME: condition (e.g., x > 0, n is integer)
- EQUATION: variable_name = expression_using_defined_variables
- SIMPLIFY: expression -> simplified_result
- SUBSTITUTE: expression_with_values -> result
- SOLVE: variable_name -> numerical_value
- CHECK: condition -> True/False
- ANSWER: final_numerical_answer

EXAMPLE 1:
Problem: A store has 40 apples. They sell 15 in the morning and 8 in the afternoon. How many are left?

DEFINE: initial = 40
DEFINE: sold_morning = 15
DEFINE: sold_afternoon = 8
EQUATION: total_sold = sold_morning + sold_afternoon
SIMPLIFY: total_sold = 15 + 8 -> 23
EQUATION: remaining = initial - total_sold
SIMPLIFY: remaining = 40 - 23 -> 17
CHECK: remaining >= 0 -> True
ANSWER: 17

EXAMPLE 2:
Problem: If 3x + 7 = 22, what is x?

DEFINE: coefficient = 3
DEFINE: constant = 7
DEFINE: target = 22
EQUATION: 3*x + 7 = 22
SIMPLIFY: 3*x = 22 - 7 -> 15
SOLVE: x -> 5
CHECK: 3*5 + 7 = 22 -> True
ANSWER: 5"""

S3MATH_USER = """Solve the following math problem using typed operation traces.

Problem: {problem}

Trace:"""

# =============================================================================
# S3-Math Repair Prompt
# =============================================================================

S3MATH_REPAIR = """Your previous trace had errors. Here is the feedback:

Problem: {problem}

Previous trace with errors:
{trace_with_errors}

Current symbolic state:
{state_snapshot}

Errors found:
{error_summary}

Please output a CORRECTED complete trace. Fix the errors while keeping correct steps.

Corrected Trace:"""

# =============================================================================
# Ablation: No Typing (free-form but with equation extraction)
# =============================================================================

ABLATION_NO_TYPING_SYSTEM = COT_SYSTEM
ABLATION_NO_TYPING_USER = """Solve the following math problem step by step. When you set up equations, write them clearly as "variable = expression". Show all arithmetic. Give the final answer as "The answer is: <number>".

Problem: {problem}

Solution:"""


def build_messages(system: str, user_template: str, problem: str) -> list[dict]:
    """Build chat messages for a given method and problem."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_template.format(problem=problem)},
    ]
