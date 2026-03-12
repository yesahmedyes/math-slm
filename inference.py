"""Model loading and generation using HuggingFace Transformers."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelInference:
    """Wraps HuggingFace Transformers for inference."""

    def __init__(self, model_name: str, max_model_len: int = 4096):
        self.model_name = model_name
        self.max_model_len = max_model_len

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def build_prompt(self, messages: list[dict], enable_thinking: bool = False) -> str:
        """Apply chat template to messages."""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def generate(
        self,
        prompts: list[str],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        n: int = 1,
        **kwargs,
    ) -> list:
        """Generate completions for a batch of prompts.

        Processes prompts one at a time to keep memory usage low.
        Returns list of outputs. Each output is either a string (n=1)
        or list of strings (n>1).
        """
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_model_len,
            ).to(self.model.device)
            prompt_len = inputs["input_ids"].shape[1]

            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
                "top_p": top_p,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature

            samples = []
            for _ in range(n):
                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, **gen_kwargs)
                text = self.tokenizer.decode(
                    output_ids[0][prompt_len:], skip_special_tokens=True
                )
                samples.append(text)

            if n == 1:
                results.append(samples[0])
            else:
                results.append(samples)
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
