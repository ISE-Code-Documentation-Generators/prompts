import torch

from transformers import AutoTokenizer
import transformers

from .main import LLM_API


class Falcon(LLM_API):
    name: str = "Falcon"

    def __init__(self) -> None:
        super().__init__()
        model = "tiiuae/falcon-7b-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
    

    def get_response(self, prompt: str) -> str:
        response = self.pipeline(
            prompt,
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        response_text = response[0]["generated_text"]
        if response_text.startswith(prompt):
            response_text = response_text[len(prompt):]

        return response_text
