"""Safe code execution sandbox for PAL and full formalization methods."""

import re
import os
import subprocess
import tempfile
from config import CODE_EXEC_TIMEOUT

ALLOWED_IMPORTS = {
    "math",
    "fractions",
    "decimal",
    "itertools",
    "functools",
    "collections",
    "sympy",
    "numpy",
    "re",
    "statistics",
}

# Each entry: (regex_pattern, human_readable_label)
DISALLOWED_PATTERNS = [
    (r"\bos\.", "os."),
    (r"\bsys\.", "sys."),
    (r"\bsubprocess\b", "subprocess"),
    (r"\bopen\s*\(", "open("),
    (r"\b__import__\s*\(", "__import__("),
    (r"(?<!\w)eval\s*\(", "eval("),
    (r"(?<!\w)exec\s*\(", "exec("),
    (r"\bcompile\s*\(", "compile("),
    (r"\bshutil\b", "shutil"),
    (r"\bpathlib\b", "pathlib"),
    (r"\bsocket\b", "socket"),
    (r"\burllib\b", "urllib"),
    (r"\brequests\b", "requests"),
    (r"\bpickle\b", "pickle"),
    (r"\bsignal\b", "signal"),
    (r"\bctypes\b", "ctypes"),
    (r"\bmultiprocessing\b", "multiprocessing"),
    (r"\bthreading\b", "threading"),
    (r"\btempfile\b", "tempfile"),
    (r"\bwebbrowser\b", "webbrowser"),
    (r"\binput\s*\(", "input("),
]


def validate_code(code: str) -> tuple:
    """Check code for disallowed operations. Returns (is_valid, reason)."""
    # Strip comments to avoid false positives
    code_no_comments = re.sub(r"#.*$", "", code, flags=re.MULTILINE)

    for pattern, label in DISALLOWED_PATTERNS:
        if re.search(pattern, code_no_comments):
            return False, f"Disallowed pattern: {label}"

    # Check imports
    import_matches = re.findall(r"(?:from|import)\s+([\w.]+)", code_no_comments)
    for imp in import_matches:
        root = imp.split(".")[0]
        if root not in ALLOWED_IMPORTS:
            return False, f"Disallowed import: {imp}"

    return True, ""


def _looks_like_code(line: str) -> bool:
    """Return True if *line* looks like a Python code line (not NL prose)."""
    s = line.strip()
    if not s:
        return True  # blank lines are neutral, keep them
    # Obvious code indicators
    if s.startswith(
        (
            "import ",
            "from ",
            "#",
            "def ",
            "class ",
            "for ",
            "if ",
            "while ",
            "return ",
            "try:",
            "with ",
            "elif ",
            "else:",
            "except",
            "finally:",
            "print",
        )
    ):
        return True
    if "=" in s or s.endswith(":") or s.endswith(")"):
        return True
    return False


def _is_nl_stop(line: str) -> bool:
    """Return True if *line* is NL prose that should stop code extraction."""
    s = line.strip()
    if not s:
        return False
    # If the line also looks like code, don't stop
    if _looks_like_code(s):
        return False
    nl_prefixes = (
        "The answer",
        "Therefore",
        "So the",
        "So,",
        "Thus ",
        "Thus,",
        "Hence ",
        "Hence,",
        "In conclusion",
        "Final answer",
        "Answer:",
        "Output:",
        "The final",
        "This gives",
        "We get",
        "Which gives",
        "That means",
        "This means",
    )
    return s.startswith(nl_prefixes)


def extract_code_from_output(text: str) -> str:
    """Extract Python code from model output, handling markdown fences."""
    # ---- Strategy 1: markdown fences (preferred) ----
    # Find ALL ```python ... ``` blocks and pick the longest
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()

    # Try ``` ... ``` (no language tag)
    blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()

    # ---- Strategy 2: strip NL preamble, then heuristic ----
    # Strip common NL preambles that chat models prepend
    cleaned = text.strip()
    preamble_patterns = [
        r"^(?:Sure!?|Okay!?|Here(?:'s| is) (?:the |my )?(?:solution|code|program|answer)[:\.]?\s*)",
        r"^(?:Let me solve this[:\.]?\s*)",
        r"^(?:I'll solve this[:\.]?\s*)",
        r"^(?:Let's solve this[:\.]?\s*)",
    ]
    for pat in preamble_patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()

    lines = cleaned.split("\n")
    code_lines = []
    started = False
    for line in lines:
        stripped = line.strip()
        if not started:
            # Look for first line that looks like Python
            if _looks_like_code(stripped) and stripped:
                started = True
                code_lines.append(line)
        else:
            # Stop if we hit NL prose (but not code-like lines)
            if _is_nl_stop(line):
                break
            code_lines.append(line)

    # Strip trailing blank lines
    while code_lines and not code_lines[-1].strip():
        code_lines.pop()

    if code_lines:
        return "\n".join(code_lines)

    return text.strip()


def execute_code(code: str) -> dict:
    """Execute Python code safely in a subprocess.

    Returns dict with keys: stdout, stderr, success, error.
    """
    is_valid, reason = validate_code(code)
    if not is_valid:
        return {
            "stdout": "",
            "stderr": reason,
            "success": False,
            "error": f"Validation failed: {reason}",
        }

    # Ensure the code has a print statement (common issue)
    if "print" not in code:
        lines = code.strip().split("\n")

        # Strategy 1: If 'ans' is assigned anywhere (PoT convention), print it
        has_ans = any(re.search(r"\bans\s*(?:=|\+=|-=|\*=|/=)", line) for line in lines)
        if has_ans:
            lines.append("print(ans)")
            code = "\n".join(lines)
        else:
            # Strategy 2: Check the last non-blank, non-comment line
            last_line = lines[-1].strip()
            if not last_line.startswith("#"):
                # Only match simple assignment: var = expr (not ==, !=, +=, etc.)
                assign_match = re.match(r"^(\w+)\s*=[^=]", last_line)
                if assign_match:
                    var_name = assign_match.group(1)
                    lines.append(f"print({var_name})")
                    code = "\n".join(lines)
                elif not last_line.startswith(("import ", "from ", "def ", "class ")):
                    # Bare expression — wrap in print
                    lines[-1] = f"print({last_line})"
                    code = "\n".join(lines)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=CODE_EXEC_TIMEOUT,
            env={
                "PATH": os.environ.get("PATH", ""),
                "HOME": "/tmp",
                "PYTHONPATH": "",
            },
        )
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "success": result.returncode == 0,
            "error": result.stderr.strip() if result.returncode != 0 else None,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "Execution timed out",
            "success": False,
            "error": f"Timed out after {CODE_EXEC_TIMEOUT}s",
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "success": False,
            "error": str(e),
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
