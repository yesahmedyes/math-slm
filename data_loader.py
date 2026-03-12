"""Dataset loading for S3-Math experiments."""

from datasets import load_dataset
from config import DATASET_CONFIGS
from answer_utils import extract_gsm8k_gold, extract_math_gold


def load_gsm8k(max_samples=None):
    """Load GSM8K test set."""
    cfg = DATASET_CONFIGS["gsm8k"]
    ds = load_dataset(cfg["path"], cfg["name"], split=cfg["split"])
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    samples = []
    for i, ex in enumerate(ds):
        gold = extract_gsm8k_gold(ex["answer"])
        samples.append({
            "idx": i,
            "problem": ex["question"],
            "gold": gold,
            "full_solution": ex["answer"],
            "source": "gsm8k",
        })
    return samples


def load_math_subset(subset_key, max_samples=None):
    """Load a filtered subset of the MATH dataset."""
    cfg = DATASET_CONFIGS[subset_key]
    ds = load_dataset(cfg["path"], split=cfg["split"])
    # Filter by type
    filter_type = cfg["filter_type"]
    filtered = [ex for ex in ds if ex["type"] == filter_type]
    if max_samples:
        filtered = filtered[:max_samples]
    samples = []
    for i, ex in enumerate(filtered):
        gold = extract_math_gold(ex["solution"])
        samples.append({
            "idx": i,
            "problem": ex["problem"],
            "gold": gold,
            "full_solution": ex["solution"],
            "source": subset_key,
            "level": ex.get("level", ""),
        })
    return samples


def load_dataset_by_name(name, max_samples=None):
    """Load a dataset by name string."""
    if name == "gsm8k":
        return load_gsm8k(max_samples)
    elif name in ("math_algebra", "math_number_theory", "math_counting_prob"):
        return load_math_subset(name, max_samples)
    elif name == "all":
        data = {}
        data["gsm8k"] = load_gsm8k(max_samples)
        data["math_algebra"] = load_math_subset("math_algebra", max_samples)
        data["math_number_theory"] = load_math_subset("math_number_theory", max_samples)
        data["math_counting_prob"] = load_math_subset("math_counting_prob", max_samples)
        return data
    else:
        raise ValueError(f"Unknown dataset: {name}")
