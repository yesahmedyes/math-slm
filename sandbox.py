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
    (r"\bos\.",              "os."),
    (r"\bsys\.",             "sys."),
    (r"\bsubprocess\b",     "subprocess"),
    (r"\bopen\s*\(",        "open("),
    (r"\b__import__\s*\(",  "__import__("),
    (r"(?<!\w)eval\s*\(",   "eval("),
    (r"(?<!\w)exec\s*\(",   "exec("),
    (r"\bcompile\s*\(",     "compile("),
    (r"\bshutil\b",         "shutil"),
    (r"\bpathlib\b",        "pathlib"),
    (r"\bsocket\b",         "socket"),
    (r"\burllib\b",         "urllib"),
    (r"\brequests\b",       "requests"),
    (r"\bpickle\b",         "pickle"),
    (r"\bsignal\b",         "signal"),
    (r"\bctypes\b",         "ctypes"),
    (r"\bmultiprocessing\b","multiprocessing"),
    (r"\bthreading\b",      "threading"),
    (r"\btempfile\b",       "tempfile"),
    (r"\bwebbrowser\b",     "webbrowser"),
    (r"\binput\s*\(",       "input("),
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


def extract_code_from_output(text: str) -> str:
    """Extract Python code from model output, handling markdown fences."""
    # Try to find code between ```python ... ```
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try ``` ... ```
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If the output starts looking like code (no markdown fences)
    lines = text.strip().split("\n")
    code_lines = []
    started = False
    for line in lines:
        stripped = line.strip()
        if not started:
            # Look for first line that looks like Python
            if (
                stripped.startswith((
                    "import ", "from ", "#", "def ", "class ",
                    "for ", "if ", "while ", "return ", "try:", "with ",
                    "elif ", "else:", "except", "finally:",
                ))
                or "=" in stripped
                or stripped.startswith("print")
                or (stripped.endswith(")") and "(" in stripped)
            ):
                started = True
                code_lines.append(line)
        else:
            # Stop if we hit something that's clearly not code
            if stripped.startswith((
                "The answer", "Therefore", "So ", "Thus ",
                "Hence ", "In conclusion", "Final answer",
                "Answer:", "Output:",
            )):
                break
            code_lines.append(line)

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
        last_line = lines[-1].strip()
        if not last_line.startswith("#"):
            # Only match simple assignment: var = expr (not ==, !=, +=, etc.)
            assign_match = re.match(r'^(\w+)\s*=[^=]', last_line)
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
