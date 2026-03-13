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

# Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
```python
money_initial = 23
bagels = 5
bagel_cost = 3
money_spent = bagels * bagel_cost
money_left = money_initial - money_spent
print(money_left)
```

# Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
```python
golf_balls_initial = 58
golf_balls_lost_tuesday = 23
golf_balls_lost_wednesday = 2
golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
print(golf_balls_left)
```

# Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
```python
computers_initial = 9
computers_per_day = 5
num_days = 4  # 4 days between monday and thursday
computers_added = computers_per_day * num_days
computers_total = computers_initial + computers_added
print(computers_total)
```

# Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
```python
cars_initial = 3
cars_arrived = 2
total_cars = cars_initial + cars_arrived
print(total_cars)
```

# Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
```python
leah_chocolates = 32
sister_chocolates = 42
total_chocolates = leah_chocolates + sister_chocolates
chocolates_eaten = 35
chocolates_left = total_chocolates - chocolates_eaten
print(chocolates_left)
```

# Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
```python
jason_lollipops_initial = 20
jason_lollipops_after = 12
denny_lollipops = jason_lollipops_initial - jason_lollipops_after
print(denny_lollipops)
```

# Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
```python
trees_initial = 15
trees_after = 21
trees_added = trees_after - trees_initial
print(trees_added)
```

# Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
```python
toys_initial = 5
mom_toys = 2
dad_toys = 2
total_received = mom_toys + dad_toys
total_toys = toys_initial + total_received
print(total_toys)
```

# Q: {problem}
"""

# =============================================================================
# Method 5: PoT (Program of Thoughts)
# =============================================================================

POT_SYSTEM = "You are a math problem solver that expresses reasoning steps as Python programs. Use meaningful variable names. You may use SymPy for symbolic math and loops for iterative problems. Store the final answer in a variable named 'ans'. Always write your code inside a ```python code block."

POT_USER = """Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
```python
# Python code, return ans
total_eggs = 16
eaten_eggs = 3
baked_eggs = 4
sold_eggs = total_eggs - eaten_eggs - baked_eggs
dollars_per_egg = 2
ans = sold_eggs * dollars_per_egg
print(ans)
```

Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts does it take in total?
```python
# Python code, return ans
bolts_of_blue_fiber = 2
bolts_of_white_fiber = bolts_of_blue_fiber / 2
ans = bolts_of_blue_fiber + bolts_of_white_fiber
print(ans)
```

Question: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?
```python
# Python code, return ans
cost_of_original_house = 80000
increase_rate = 150 / 100
value_of_house = (1 + increase_rate) * cost_of_original_house
cost_of_repair = 50000
ans = value_of_house - cost_of_repair - cost_of_original_house
print(ans)
```

Question: Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?
```python
# Python code, return ans
numb_of_chickens = 20
cups_for_each_chicken = 3
cups_for_all_chicken = numb_of_chickens * cups_for_each_chicken
cups_in_the_morning = 15
cups_in_the_afternoon = 25
ans = cups_for_all_chicken - cups_in_the_morning - cups_in_the_afternoon
print(ans)
```

Question: Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?
```python
# Python code, return ans
num_glasses = 16
first_glass_cost = 5
second_glass_cost = 5 * 0.6
ans = 0
for i in range(num_glasses):
    if i % 2 == 0:
        ans += first_glass_cost
    else:
        ans += second_glass_cost
print(ans)
```

Question: Marissa is hiking a 12-mile trail. She took 1 hour to walk the first 4 miles, then another hour to walk the next two miles. If she wants her average speed to be 4 miles per hour, what speed (in miles per hour) does she need to walk the remaining distance?
```python
# Python code, return ans
average_mile_per_hour = 4
total_trail_miles = 12
remaining_miles = total_trail_miles - 4 - 2
total_hours = total_trail_miles / average_mile_per_hour
remaining_hours = total_hours - 2
ans = remaining_miles / remaining_hours
print(ans)
```

Question: Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it will grow 7 lemons, which he can sell for $1.5 each. It costs $3 a year to water and feed the tree. How many years will it take before he starts earning money on the lemon tree?
```python
# Python code, return ans
total_cost = 90
cost_of_watering_and_feeding = 3
cost_of_each_lemon = 1.5
num_of_lemon_per_year = 7
ans = 0
while total_cost > 0:
    total_cost += cost_of_watering_and_feeding
    total_cost -= num_of_lemon_per_year * cost_of_each_lemon
    ans += 1
print(ans)
```

Question: When Freda cooks canned tomatoes into sauce, they lose half their volume. Each 16 ounce can of tomatoes that she uses contains three tomatoes. Freda's last batch of tomato sauce made 32 ounces of sauce. How many tomatoes did Freda use?
```python
# Python code, return ans
lose_rate = 0.5
num_tomato_contained_in_per_ounce_sauce = 3 / 16
ounce_sauce_in_last_batch = 32
num_tomato_in_last_batch = ounce_sauce_in_last_batch * num_tomato_contained_in_per_ounce_sauce
ans = num_tomato_in_last_batch / (1 - lose_rate)
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
