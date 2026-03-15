"""Central configuration for S3-Math experiments."""

MODELS = {
    "4B": "Qwen/Qwen3.5-4B",
    "2B": "Qwen/Qwen3.5-2B",
    "0.8B": "Qwen/Qwen3.5-0.8B",
}

# Generation parameters per method
SAMPLING_CONFIGS = {
    "direct": {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256,
    },
    "cot": {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1024,
    },
    "sc_cot": {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 1024,
        "n": 5,
    },
    "pal": {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1024,
    },
    "pot": {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1024,
    },
    "formal": {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1024,
    },
    "s3math": {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1024,
    },
}

# Dataset configurations
DATASET_CONFIGS = {
    "gsm8k": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "test",
    },
    "math_algebra": {
        "path": "EleutherAI/hendrycks_math",
        "name": "algebra",
        "split": "test",
    },
    "math_number_theory": {
        "path": "EleutherAI/hendrycks_math",
        "name": "number_theory",
        "split": "test",
    },
    "math_counting_prob": {
        "path": "EleutherAI/hendrycks_math",
        "name": "counting_and_probability",
        "split": "test",
    },
}

TASK_CATEGORIES = {
    "basic_arithmetic": ["gsm8k"],
    "algebra_word": ["math_algebra"],
    "constraint_heavy": ["math_number_theory", "math_counting_prob"],
}

# S3-Math specific
S3MATH_MAX_REPAIRS = 2
S3MATH_SYMPY_TIMEOUT = 5  # seconds per operation
CODE_EXEC_TIMEOUT = 10  # seconds for code execution
