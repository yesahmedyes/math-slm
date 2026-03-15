"""Prompt templates for all 10 evaluation methods."""


def build_messages(system: str, user_template: str, problem: str) -> list[dict]:
    """Build a chat message list from system prompt, user template, and problem."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_template.format(problem=problem)},
    ]


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

PAL_SYSTEM = "You are a math problem solver that decomposes problems into Python programs with meaningful variable names. Always write your code inside a ```python code block. Begin with a '# Reasoning:' comment explaining your approach, then add step comments."

PAL_USER = """Let's use Python to solve math problems. Use meaningful variable names that reflect the problem entities. Write the question as a comment starting with # Q:, then the Python code inside a ```python code block, ending with a print statement.

# Q: A tech support agent receives 45 tickets each day. She resolves 38 tickets per day. After 5 working days, how many unresolved tickets has she accumulated?
```python
# Reasoning: Find daily unresolved = received - resolved, then scale by number of days.
tickets_received_per_day = 45
tickets_resolved_per_day = 38
days_worked = 5
# Step 1: daily unresolved tickets
unresolved_per_day = tickets_received_per_day - tickets_resolved_per_day
# Step 2: total over 5 days
total_unresolved = unresolved_per_day * days_worked
print(total_unresolved)
```

# Q: A laptop originally costs $250. The store offers a 20% discount, but then charges 8% sales tax on the discounted price. What is the final price in dollars?
```python
# Reasoning: Apply discount first to get sale price, then compute tax on that price.
original_price = 250
discount_percent = 20
tax_percent = 8
# Step 1: apply discount
discounted_price = original_price * (1 - discount_percent / 100)
# Step 2: add sales tax on discounted price
sales_tax = discounted_price * tax_percent / 100
final_price = discounted_price + sales_tax
print(final_price)
```

# Q: Solve for x: 3x + 7 = 2x + 15. What is the value of x?
```python
# Reasoning: Move x terms to left side and constants to right, then solve for x.
# 3x - 2x = 15 - 7 => x = 8
coefficient_left = 3
coefficient_right = 2
constant_left = 7
constant_right = 15
# Step 1: isolate x by rearranging
x = (constant_right - constant_left) // (coefficient_left - coefficient_right)
print(x)
```

# Q: Let f(x) = 2x^2 - 3x + 1. What is f(4) - f(2)?
```python
# Reasoning: Define f(x), evaluate at both points, then subtract.
def f(x):
    return 2 * x**2 - 3 * x + 1
# Step 1: compute f(4) - f(2)
result = f(4) - f(2)
print(result)
```

# Q: Two buses depart from the same station. Bus A departs every 48 minutes and Bus B departs every 180 minutes. If they both depart at 6:00 AM, how many minutes after 6:00 AM will they next depart at the same time?
```python
# Reasoning: They next coincide at LCM(48, 180). Use LCM = a*b / GCD(a,b).
import math
bus_a_interval = 48
bus_b_interval = 180
# Step 1: compute GCD, then derive LCM
gcd = math.gcd(bus_a_interval, bus_b_interval)
lcm = bus_a_interval * bus_b_interval // gcd
print(lcm)
```

# Q: What is the remainder when 7^100 is divided by 5?
```python
# Reasoning: Use Python's built-in modular exponentiation: pow(base, exp, mod).
base = 7
exponent = 100
modulus = 5
# Step 1: compute 7^100 mod 5 directly
remainder = pow(base, exponent, modulus)
print(remainder)
```

# Q: A club has 8 members. How many ways can they choose a committee of 3 members?
```python
# Reasoning: Order doesn't matter, so use combinations: C(8,3) = 8! / (3! * 5!).
import math
total_members = 8
committee_size = 3
# Step 1: compute C(8,3)
ways = math.factorial(total_members) // (math.factorial(committee_size) * math.factorial(total_members - committee_size))
print(ways)
```

# Q: Two fair six-sided dice are rolled. What is the probability that their sum is at least 10? Express your answer as a common fraction.
```python
# Reasoning: Enumerate all 36 outcomes of two dice, count those with sum >= 10.
from fractions import Fraction
favorable = 0
total = 0
# Step 1: count favorable outcomes
for die1 in range(1, 7):
    for die2 in range(1, 7):
        total += 1
        if die1 + die2 >= 10:
            favorable += 1
# Step 2: form probability as fraction
probability = Fraction(favorable, total)
print(probability)
```

# Q: {problem}
"""

# =============================================================================
# Method 5: PoT (Program of Thoughts)
# =============================================================================

POT_SYSTEM = "You are a math problem solver that expresses reasoning steps as Python programs. Use meaningful variable names. You may use SymPy for symbolic math and loops for iterative problems. Store the final answer in a variable named 'ans'. Always write your code inside a ```python code block. Begin with a '# Reasoning:' comment explaining your approach, then add step comments."

POT_USER = """Question: Maria is reading a 320-page book. On weekdays she reads 25 pages per day, and on weekends she reads 40 pages per day. How many full weeks does it take her to finish the book?
```python
# Python code, return ans
# Reasoning: Compute pages per week (5 weekdays + 2 weekend days), then ceil(total / per_week).
import math
pages_total = 320
pages_weekday = 25
pages_weekend = 40
weekdays = 5
weekend_days = 2
# Step 1: weekly reading rate
pages_per_week = pages_weekday * weekdays + pages_weekend * weekend_days
# Step 2: weeks needed (round up)
ans = math.ceil(pages_total / pages_per_week)
print(ans)
```

Question: A swimming pool holds 480 gallons. A hose fills it at 12 gallons per hour, but a small crack drains water at 4 gallons per hour. How many hours does it take to fill the pool from empty?
```python
# Python code, return ans
# Reasoning: Net fill rate = fill - drain. Time = capacity / net rate.
pool_capacity = 480
fill_rate = 12
drain_rate = 4
# Step 1: net fill rate
net_fill_rate = fill_rate - drain_rate
# Step 2: hours to fill
ans = pool_capacity // net_fill_rate
print(ans)
```

Question: If 2x + y = 11 and x - y = 1, what is the value of x + y?
```python
# Python code, return ans
# Reasoning: Two equations, two unknowns — use sympy.solve for the system.
from sympy import symbols, Eq, solve
x, y = symbols('x y')
# Step 1: set up equations
eq1 = Eq(2*x + y, 11)
eq2 = Eq(x - y, 1)
# Step 2: solve and compute x + y
solution = solve((eq1, eq2), (x, y))
ans = solution[x] + solution[y]
print(ans)
```

Question: Find the positive difference between the roots of x^2 - 5x + 6 = 0.
```python
# Python code, return ans
# Reasoning: Find both roots of the quadratic, then take their absolute difference.
from sympy import symbols, solve
x = symbols('x')
# Step 1: solve for roots
roots = solve(x**2 - 5*x + 6, x)
# Step 2: positive difference
ans = max(roots) - min(roots)
print(ans)
```

Question: What is the sum of all positive divisors of 72?
```python
# Python code, return ans
# Reasoning: A divisor of n is any integer i where n % i == 0. Sum all such i from 1 to n.
n = 72
# Step 1: find and sum all divisors
ans = sum(i for i in range(1, n + 1) if n % i == 0)
print(ans)
```

Question: What is the smallest positive integer that is divisible by both 12 and 18 and is also a perfect square?
```python
# Python code, return ans
# Reasoning: Start with LCM(12,18). For a perfect square, all prime exponents must be even.
# Multiply by primes whose exponents are odd.
from sympy import factorint, lcm
l = int(lcm(12, 18))
factors = factorint(l)
# Step 1: fix odd exponents
multiplier = 1
for prime, exp in factors.items():
    if exp % 2 == 1:
        multiplier *= prime
# Step 2: multiply to get smallest perfect square
ans = l * multiplier
print(ans)
```

Question: In how many ways can 5 different books be arranged on a shelf if two specific books must always be next to each other?
```python
# Python code, return ans
# Reasoning: Treat the two paired books as one unit → 4 units to arrange.
# The pair has 2! internal orderings.
import math
# Step 1: arrange 4 units, multiply by internal orderings
ans = math.factorial(4) * math.factorial(2)
print(ans)
```

Question: A bag contains 5 red marbles and 3 blue marbles. If you draw 3 marbles without replacement, what is the probability of getting exactly 2 red marbles? Express your answer as a common fraction.
```python
# Python code, return ans
# Reasoning: Use hypergeometric probability: C(red,2)*C(blue,1) / C(total,3).
from math import comb
from fractions import Fraction
red_marbles = 5
blue_marbles = 3
draw_count = 3
# Step 1: count favorable and total outcomes
favorable = comb(red_marbles, 2) * comb(blue_marbles, 1)
total = comb(red_marbles + blue_marbles, draw_count)
# Step 2: form probability
ans = Fraction(favorable, total)
print(ans)
```

Question: {problem}
"""

# =============================================================================
# Method 6: CoMAT (Chain of Mathematically Annotated Thought)
# =============================================================================

COMAT_SYSTEM = """You are a math problem solver that converts problems into symbolic representations and solves them using strict logical reasoning. Follow the structured steps precisely."""

COMAT_USER = """You are given a mathematical question. Your task is to convert it into a symbolic representation and solve it using strict logical reasoning, without any reliance on memorized answers or external knowledge. Rigorously follow these steps:

1. Identify and define all the predicates, functions, and variables in the problem.
2. Parse the entire problem into logical rules based strictly on the defined predicates, functions, and variables.
3. Write all the facts explicitly mentioned in the problem as logical statements.
4. Parse the question into a symbolic form using only the defined predicates and variables.
5. Solve the problem step by step using only the symbolic representations and logical reasoning. Provide clear reasoning for each step.
6. Derive the final answer based solely on your symbolic solution and step-by-step reasoning. DO NOT INCLUDE any symbols or units, just the number.

IMPORTANT:
- Base your entire solution on the symbolic representation and logical reasoning.
- Do not rely on any prior knowledge, memorized answers, or previous examples.

After completing the symbolic representation and reasoning, provide the final answer on the last line as "The answer is: <number>".

Here is an example:

Question: Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?

1. Define predicates and functions:
- Earnings(x): Function that calculates total earnings for x hours worked
- RegularPayRate: Constant representing Eliza's regular hourly rate ($10)
- OvertimePayRate: Constant representing Eliza's overtime hourly rate (1.2 * RegularPayRate)
- HoursWorked: Variable representing the total hours worked (45)
- RegularHours: Constant representing the number of regular hours (40)
- OvertimeHours: Variable representing the number of overtime hours (HoursWorked - RegularHours)

2. Parse problem into logical rules:
- For all x: Earnings(x) = (RegularPayRate * RegularHours) + (OvertimePayRate * OvertimeHours)
- OvertimePayRate = 1.2 * RegularPayRate

3. Facts:
- RegularPayRate = $10
- HoursWorked = 45
- RegularHours = 40

4. Parse the question:
- Find Earnings(HoursWorked)

5. Solve step by step:
(a) Calculate OvertimePayRate:
    OvertimePayRate = 1.2 * RegularPayRate = 1.2 * $10 = $12
(b) Calculate OvertimeHours:
    OvertimeHours = HoursWorked - RegularHours = 45 - 40 = 5
(c) Calculate Earnings(HoursWorked):
    Earnings(HoursWorked) = (RegularPayRate * RegularHours) + (OvertimePayRate * OvertimeHours)
                          = ($10 * 40) + ($12 * 5)
                          = $400 + $60
                          = $460

6. Derived answer:
Eliza's earnings for this week are $460.

The answer is: 460

Now solve the following problem using the same structured approach:

Question: {problem}
"""

# =============================================================================
# Method 7: Logic-LM (LLM + Symbolic Solver with Self-Refinement)
# =============================================================================

LOGIC_LM_SYSTEM = """You are a math problem solver that formulates problems as structured symbolic programs using SymPy. Your task is to REPRESENT the problem symbolically and let the solver compute the answer. Always write your code inside a ```python code block."""

LOGIC_LM_USER = """You are given a math problem. Your task is to:
1) Define all symbolic variables used in the problem
2) Set up the equations or expressions that capture the problem's constraints
3) Use SymPy's solver to compute the answer
4) Print the final numerical result

Write ONLY a SymPy program inside a ```python code block. Follow this structured format:
- Symbols: declare all symbolic variables
- Equations: set up equations/expressions from the problem
- Solve: call sympy.solve, sympy.simplify, or evaluate as needed
- Result: print the final numerical answer

Problem: A store sells notebooks for $4 each and pens for $1 each. If a student buys 5 more pens than notebooks and spends $35 in total, how many notebooks did the student buy?
```python
from sympy import symbols, Eq, solve
# Symbols
n, p = symbols('n p')
# Equations: p = n + 5 (5 more pens than notebooks), 4n + 1p = 35 (total cost)
eq1 = Eq(p, n + 5)
eq2 = Eq(4*n + 1*p, 35)
# Solve
solution = solve((eq1, eq2), (n, p))
# Result
print(solution[n])
```

Problem: What is the sum of all integers from 1 to 100 that are divisible by 3 but not by 5?
```python
from sympy import Symbol, summation, floor
# Symbols: use direct computation since this is a finite sum with conditions
# Equations: sum of integers in [1,100] divisible by 3 but not 5
result = sum(i for i in range(1, 101) if i % 3 == 0 and i % 5 != 0)
# Result
print(result)
```

Problem: Find all values of x such that x^2 - 7x + 12 = 0. What is the larger root?
```python
from sympy import symbols, solve
# Symbols
x = symbols('x')
# Equations
roots = solve(x**2 - 7*x + 12, x)
# Result
print(max(roots))
```

Problem: A triangle has sides of length 5, 12, and 13. What is its area?
```python
from sympy import sqrt, Rational
# Symbols: use Heron's formula with known side lengths
a, b, c = 5, 12, 13
# Equations: semi-perimeter and Heron's formula
s = Rational(a + b + c, 2)
area = sqrt(s * (s - a) * (s - b) * (s - c))
# Result
print(area)
```

Problem: How many ways can 6 people be seated in a row if two specific people must NOT sit next to each other?
```python
from sympy import factorial
# Symbols: total arrangements minus arrangements where the two ARE adjacent
total_people = 6
# Equations: total = 6!, adjacent = treat pair as unit -> 5! * 2!
total_arrangements = factorial(total_people)
adjacent_arrangements = factorial(total_people - 1) * factorial(2)
# Solve: subtract to get non-adjacent
result = total_arrangements - adjacent_arrangements
# Result
print(result)
```

Problem: If 3^x = 81, what is the value of 3^(x-2)?
```python
from sympy import symbols, Eq, solve
# Symbols
x = symbols('x')
# Equations: 3^x = 81
eq = Eq(3**x, 81)
x_val = solve(eq, x)[0]
# Solve: compute 3^(x-2)
result = 3**(x_val - 2)
# Result
print(result)
```

Problem: Two fair coins are flipped. What is the probability that at least one shows heads? Express as a fraction.
```python
from sympy import Rational
# Symbols: P(at least one head) = 1 - P(no heads)
# Equations: P(no heads) = P(TT) = 1/2 * 1/2 = 1/4
p_no_heads = Rational(1, 2) * Rational(1, 2)
# Solve
p_at_least_one_head = 1 - p_no_heads
# Result
print(p_at_least_one_head)
```

Problem: What is the remainder when 2^50 is divided by 7?
```python
from sympy import Mod, Integer
# Symbols: compute 2^50 mod 7
# Solve: use modular exponentiation
result = Mod(Integer(2)**50, 7)
# Result
print(result)
```

Problem: {problem}
"""

LOGIC_LM_REFINE_SYSTEM = """You are a math problem solver that fixes symbolic programs. You will be given a SymPy program that produced an error. Fix the program based on the error message. Always write your corrected code inside a ```python code block."""

LOGIC_LM_REFINE_USER = """The following SymPy program for the math problem produced an error. Fix the program.

Problem: {problem}

Previous code:
```python
{code}
```

Error message:
{error}

Write a corrected version inside a ```python code block. Keep the same structured format (Symbols, Equations, Solve, Result) and ensure it prints the final numerical answer."""

# =============================================================================
# Method 8: ToRA (Tool-integrated Reasoning Agent)
# =============================================================================

TORA_SYSTEM = """Integrate step-by-step reasoning and Python code to solve math problems using the following guidelines:
- Analyze the question and write functions to solve the problem; the function should not take any arguments.
- You may use libraries like sympy, math, and fractions.
- Interleave natural language reasoning with code. First reason about the approach, then write code to compute, then interpret results and continue reasoning if needed.
- Always write code inside ```python code blocks.
- After each code block, you will receive the execution output inside ```output blocks.
- Once you have the final answer, state it clearly as "The answer is: <number>"."""

TORA_USER = """Integrate step-by-step reasoning and Python code to solve the following math problem.

---

Question: A bakery makes 120 muffins each morning. They sell 68 muffins before noon and another 32 muffins in the afternoon. The remaining unsold muffins are donated to a shelter. Over a 6-day work week, how many muffins does the bakery donate in total?

Solution:
We need to find the number of unsold muffins per day and multiply by 6 days.

```python
def donated_muffins():
    muffins_made = 120
    sold_morning = 68
    sold_afternoon = 32
    unsold_per_day = muffins_made - sold_morning - sold_afternoon
    days = 6
    total_donated = unsold_per_day * days
    return total_donated

result = donated_muffins()
print(result)
```
```output
120
```

The bakery donates 20 muffins per day, and over 6 days that is 120 muffins.

The answer is: 120

---

Question: If 2x + y = 11 and x - y = 1, what is the value of x + y?

Solution:
We have a system of two linear equations with two unknowns. Let's solve it using SymPy.

```python
from sympy import symbols, Eq, solve

def solve_system():
    x, y = symbols('x y')
    eq1 = Eq(2*x + y, 11)
    eq2 = Eq(x - y, 1)
    solution = solve((eq1, eq2), (x, y))
    return solution[x] + solution[y]

result = solve_system()
print(result)
```
```output
7
```

From the system of equations, we get x = 4 and y = 3, so x + y = 7.

The answer is: 7

---

Question: The difference between two perfect squares is 133. What is the smallest possible sum of the two perfect squares?

Solution:
The difference of two squares can be factored as x^2 - y^2 = (x+y)(x-y) = 133.
We need to find factor pairs of 133 and check which gives the smallest sum of squares.

```python
def find_factor_pairs():
    # Find all factor pairs of 133 where both factors have the same parity
    # (needed for x, y to be integers)
    n = 133
    pairs = []
    for i in range(1, n + 1):
        if n % i == 0:
            j = n // i
            if j >= i and (i + j) % 2 == 0:
                x = (i + j) // 2
                y = (j - i) // 2
                pairs.append((x, y, x**2 + y**2))
    return pairs

pairs = find_factor_pairs()
for x, y, s in pairs:
    print(f"x={x}, y={y}, x^2+y^2={s}")
```
```output
x=13, y=6, x^2+y^2=205
x=67, y=66, x^2+y^2=8845
```

We have two valid pairs: (13, 6) and (67, 66). The pair (13, 6) gives the smallest sum of squares.

Let's verify: 13^2 - 6^2 = 169 - 36 = 133 ✓

```python
def verify():
    return 13**2 + 6**2

result = verify()
print(result)
```
```output
205
```

The smallest possible sum of the two perfect squares is 205.

The answer is: 205

---

Question: What is the sum of all positive divisors of 72?

Solution:
We need to find every positive integer that divides 72 evenly and sum them. Let's first find the prime factorization, then compute the divisor sum.

```python
def sum_of_divisors():
    n = 72
    divisors = [i for i in range(1, n + 1) if n % i == 0]
    return divisors, sum(divisors)

divisors, total = sum_of_divisors()
print(f"Divisors: {divisors}")
print(f"Sum: {total}")
```
```output
Divisors: [1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72]
Sum: 195
```

The 12 positive divisors of 72 are 1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, and 72. Their sum is 195.

The answer is: 195

---

Question: A bag contains 5 red marbles and 3 blue marbles. If you draw 3 marbles without replacement, what is the probability of getting exactly 2 red and 1 blue marble? Express your answer as a common fraction.

Solution:
This is a hypergeometric probability problem. We need C(5,2) * C(3,1) / C(8,3).

```python
from math import comb
from fractions import Fraction

def probability():
    red = 5
    blue = 3
    total = red + blue
    draw = 3
    favorable = comb(red, 2) * comb(blue, 1)
    total_outcomes = comb(total, draw)
    return Fraction(favorable, total_outcomes)

result = probability()
print(result)
```
```output
15/28
```

C(5,2) = 10, C(3,1) = 3, so favorable = 30. C(8,3) = 56. The probability is 30/56 = 15/28.

The answer is: 15/28

---

Question: {problem}

Solution:
"""

TORA_CONTINUE_USER = """{output_block}
"""
