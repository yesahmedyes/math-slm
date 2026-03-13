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
# Method 4: PAL (Program-Aided Language Models)
# =============================================================================

PAL_SYSTEM = "You are a math problem solver that decomposes problems into Python programs with meaningful variable names."

PAL_USER = """Let's use Python to solve math problems. Use meaningful variable names that reflect the problem entities. Write the question as a comment starting with # Q:, then the Python code, ending with a print statement.

# Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
money_initial = 23
bagels = 5
bagel_cost = 3
money_spent = bagels * bagel_cost
money_left = money_initial - money_spent
print(money_left)

# Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
golf_balls_initial = 58
golf_balls_lost_tuesday = 23
golf_balls_lost_wednesday = 2
golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
print(golf_balls_left)

# Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
computers_initial = 9
computers_per_day = 5
num_days = 4  # 4 days between monday and thursday
computers_added = computers_per_day * num_days
computers_total = computers_initial + computers_added
print(computers_total)

# Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
cars_initial = 3
cars_arrived = 2
total_cars = cars_initial + cars_arrived
print(total_cars)

# Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
leah_chocolates = 32
sister_chocolates = 42
total_chocolates = leah_chocolates + sister_chocolates
chocolates_eaten = 35
chocolates_left = total_chocolates - chocolates_eaten
print(chocolates_left)

# Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
jason_lollipops_initial = 20
jason_lollipops_after = 12
denny_lollipops = jason_lollipops_initial - jason_lollipops_after
print(denny_lollipops)

# Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
trees_initial = 15
trees_after = 21
trees_added = trees_after - trees_initial
print(trees_added)

# Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
toys_initial = 5
mom_toys = 2
dad_toys = 2
total_received = mom_toys + dad_toys
total_toys = toys_initial + total_received
print(total_toys)

# Q: {problem}
"""

# =============================================================================
# Method 5: Full Formalization (SymPy)
# =============================================================================

FORMAL_SYSTEM = (
    "You are a math problem solver that uses SymPy for symbolic computation."
)

FORMAL_USER = """Solve the following math problem by writing a complete SymPy program.
Write your solution inside a ```python code block.
Use `from sympy import *` at the top. Define all variables as symbols, set up all equations, solve them symbolically, and print ONLY the final numerical answer.

Problem: {problem}"""

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
