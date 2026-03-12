"""Safe code execution sandbox for code-only and full formalization methods."""

import re
import os
import subprocess
import tempfile
from config import CODE_EXEC_TIMEOUT

ALLOWED_IMPORTS = {
    "math", "fractions", "decimal", "itertools", "functools",
    "collections", "sympy", "numpy", "re", "statistics",
}

DISALLOWED_PATTERNS = [
    "os.", "sys.", "subprocess", "open(", "__import__",
    "eval(", "exec(", "compile(", "shutil", "pathlib",
    "socket", "http", "urllib", "requests", "pickle",
    "signal", "ctypes", "multiprocessing", "threading",
    "glob", "tempfile", "webbrowser", "input(",
]


def validate_code(code: str) -> tuple:
    """Check code for disallowed operations. Returns (is_valid, reason)."""
    for pattern in DISALLOWED_PATTERNS:
        if pattern in code:
            return False, f"Disallowed pattern: {pattern}"

    # Check imports
    import_matches = re.findall(r"(?:from|import)\s+([\w.]+)", code)
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
            if (stripped.startswith(("import ", "from ", "#", "def ", "class "))
                    or "=" in stripped
                    or stripped.startswith("print")):
                started = True
                code_lines.append(line)
        else:
            # Stop if we hit something that's clearly not code
            if stripped.startswith(("The answer", "Therefore", "So ")):
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
        # Try to add print for the last expression
        lines = code.strip().split("\n")
        last_line = lines[-1].strip()
        if "=" in last_line and not last_line.startswith("#"):
            var_name = last_line.split("=")[0].strip()
            lines.append(f"print({var_name})")
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
