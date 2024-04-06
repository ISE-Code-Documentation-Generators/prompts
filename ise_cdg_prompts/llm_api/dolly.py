
import torch
from transformers import pipeline

from .main import LLM_API


class Dolly(LLM_API):
    default_dolly_kwargs = dict(
        model="databricks/dolly-v2-3b",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.__dolly = pipeline(**self.default_dolly_kwargs, **kwargs)

    def get_response(self, prompt: str) -> str:
        dolly_response = self.__dolly(prompt, max_new_tokens=100)
        return dolly_response[0]["generated_text"]

