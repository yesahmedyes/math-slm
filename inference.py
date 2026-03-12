"""Model loading and batched generation using vLLM."""

from vllm import LLM, SamplingParams


class ModelInference:
    """Wraps vLLM for efficient batched inference."""

    def __init__(self, model_name: str, max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.90):
        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def build_prompt(self, messages: list[dict], enable_thinking: bool = False) -> str:
        """Apply chat template to messages."""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def generate(self, prompts: list[str], temperature: float = 0.0,
                 top_p: float = 1.0, max_tokens: int = 2048,
                 n: int = 1, **kwargs) -> list:
        """Generate completions for a batch of prompts.

        Returns list of outputs. Each output is either a string (n=1)
        or list of strings (n>1).
        """
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
        )
        outputs = self.llm.generate(prompts, params)

        results = []
        for output in outputs:
            if n == 1:
                results.append(output.outputs[0].text)
            else:
                results.append([o.text for o in output.outputs])
        return results

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self.tokenizer.encode(text))


def load_model(model_key: str, **kwargs) -> ModelInference:
    """Load a model by short key name (0.8B, 2B, 4B)."""
    from config import MODELS
    model_name = MODELS[model_key]
    print(f"Loading model: {model_name}")
    return ModelInference(model_name, **kwargs)
