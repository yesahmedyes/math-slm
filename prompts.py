"""Prompt templates for all 7 evaluation methods."""

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

PAL_SYSTEM = "You are a math problem solver that decomposes problems into Python programs with meaningful variable names. Always write your code inside a ```python code block."

PAL_USER = """Let's use Python to solve math problems. Use meaningful variable names that reflect the problem entities. Write the question as a comment starting with # Q:, then the Python code inside a ```python code block, ending with a print statement.

# Q: A tech support agent receives 45 tickets each day. She resolves 38 tickets per day. After 5 working days, how many unresolved tickets has she accumulated?
```python
tickets_received_per_day = 45
tickets_resolved_per_day = 38
days_worked = 5
unresolved_per_day = tickets_received_per_day - tickets_resolved_per_day
total_unresolved = unresolved_per_day * days_worked
print(total_unresolved)
```

# Q: A laptop originally costs $250. The store offers a 20% discount, but then charges 8% sales tax on the discounted price. What is the final price in dollars?
```python
original_price = 250
discount_percent = 20
tax_percent = 8
discounted_price = original_price * (1 - discount_percent / 100)
sales_tax = discounted_price * tax_percent / 100
final_price = discounted_price + sales_tax
print(final_price)
```

# Q: Solve for x: 3x + 7 = 2x + 15. What is the value of x?
```python
# 3x + 7 = 2x + 15
# Rearrange: 3x - 2x = 15 - 7
coefficient_left = 3
coefficient_right = 2
constant_left = 7
constant_right = 15
x = (constant_right - constant_left) // (coefficient_left - coefficient_right)
print(x)
```

# Q: Let f(x) = 2x^2 - 3x + 1. What is f(4) - f(2)?
```python
def f(x):
    return 2 * x**2 - 3 * x + 1
result = f(4) - f(2)
print(result)
```

# Q: Two buses depart from the same station. Bus A departs every 48 minutes and Bus B departs every 180 minutes. If they both depart at 6:00 AM, how many minutes after 6:00 AM will they next depart at the same time?
```python
import math
bus_a_interval = 48
bus_b_interval = 180
gcd = math.gcd(bus_a_interval, bus_b_interval)
lcm = bus_a_interval * bus_b_interval // gcd
print(lcm)
```

# Q: What is the remainder when 7^100 is divided by 5?
```python
base = 7
exponent = 100
modulus = 5
remainder = pow(base, exponent, modulus)
print(remainder)
```

# Q: A club has 8 members. How many ways can they choose a committee of 3 members?
```python
import math
total_members = 8
committee_size = 3
ways = math.factorial(total_members) // (math.factorial(committee_size) * math.factorial(total_members - committee_size))
print(ways)
```

# Q: Two fair six-sided dice are rolled. What is the probability that their sum is at least 10? Express your answer as a common fraction.
```python
from fractions import Fraction
favorable = 0
total = 0
for die1 in range(1, 7):
    for die2 in range(1, 7):
        total += 1
        if die1 + die2 >= 10:
            favorable += 1
probability = Fraction(favorable, total)
print(probability)
```

# Q: {problem}
"""

# =============================================================================
# Method 5: PoT (Program of Thoughts)
# =============================================================================

POT_SYSTEM = "You are a math problem solver that expresses reasoning steps as Python programs. Use meaningful variable names. You may use SymPy for symbolic math and loops for iterative problems. Store the final answer in a variable named 'ans'. Always write your code inside a ```python code block."

POT_USER = """Question: Maria is reading a 320-page book. On weekdays she reads 25 pages per day, and on weekends she reads 40 pages per day. How many full weeks does it take her to finish the book?
```python
# Python code, return ans
import math
pages_total = 320
pages_weekday = 25
pages_weekend = 40
weekdays = 5
weekend_days = 2
pages_per_week = pages_weekday * weekdays + pages_weekend * weekend_days
ans = math.ceil(pages_total / pages_per_week)
print(ans)
```

Question: A swimming pool holds 480 gallons. A hose fills it at 12 gallons per hour, but a small crack drains water at 4 gallons per hour. How many hours does it take to fill the pool from empty?
```python
# Python code, return ans
pool_capacity = 480
fill_rate = 12
drain_rate = 4
net_fill_rate = fill_rate - drain_rate
ans = pool_capacity // net_fill_rate
print(ans)
```

Question: If 2x + y = 11 and x - y = 1, what is the value of x + y?
```python
# Python code, return ans
from sympy import symbols, Eq, solve
x, y = symbols('x y')
eq1 = Eq(2*x + y, 11)
eq2 = Eq(x - y, 1)
solution = solve((eq1, eq2), (x, y))
ans = solution[x] + solution[y]
print(ans)
```

Question: Find the positive difference between the roots of x^2 - 5x + 6 = 0.
```python
# Python code, return ans
from sympy import symbols, solve
x = symbols('x')
roots = solve(x**2 - 5*x + 6, x)
ans = max(roots) - min(roots)
print(ans)
```

Question: What is the sum of all positive divisors of 72?
```python
# Python code, return ans
n = 72
ans = sum(i for i in range(1, n + 1) if n % i == 0)
print(ans)
```

Question: What is the smallest positive integer that is divisible by both 12 and 18 and is also a perfect square?
```python
# Python code, return ans
from sympy import factorint, lcm
l = int(lcm(12, 18))
factors = factorint(l)
multiplier = 1
for prime, exp in factors.items():
    if exp % 2 == 1:
        multiplier *= prime
ans = l * multiplier
print(ans)
```

Question: In how many ways can 5 different books be arranged on a shelf if two specific books must always be next to each other?
```python
# Python code, return ans
import math
# Treat the two books as a single unit: 4 units to arrange
# The two books can be ordered 2! ways within the unit
ans = math.factorial(4) * math.factorial(2)
print(ans)
```

Question: A bag contains 5 red marbles and 3 blue marbles. If you draw 3 marbles without replacement, what is the probability of getting exactly 2 red marbles? Express your answer as a common fraction.
```python
# Python code, return ans
from math import comb
from fractions import Fraction
red_marbles = 5
blue_marbles = 3
draw_count = 3
favorable = comb(red_marbles, 2) * comb(blue_marbles, 1)
total = comb(red_marbles + blue_marbles, draw_count)
ans = Fraction(favorable, total)
print(ans)
```

Question: {problem}
"""

# =============================================================================
# Method 6: Full Formalization (SymPy)
# =============================================================================

FORMAL_SYSTEM = (
    "You are a math problem solver that uses SymPy for symbolic computation."
)

FORMAL_USER = """Solve the following math problem by writing a complete SymPy program.
Write your solution inside a ```python code block.
Use `from sympy import *` at the top. Define all variables as symbols, set up all equations, solve them symbolically, and print ONLY the final numerical answer.

Problem: {problem}"""

# =============================================================================
# Method 7: S3-Math (Selective Neuro-Symbolic State Scaffolding)
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
