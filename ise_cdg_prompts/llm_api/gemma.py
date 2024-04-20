# get_ipython().system('pip install -U "transformers==4.38.1" --upgrade')
# get_ipython().system("pip install accelerate")
# get_ipython().system("pip install -i https://pypi.org/simple/ bitsandbytes")

from ise_cdg_prompts.llm_api.main import LLM_API


class Gemma(LLM_API):
    name: str = "Gemma"
    def __init__(self) -> None:
        super().__init__()
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import os

        os.environ["HF_TOKEN"] = "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"

        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
        self.model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2b-it", device_map="auto"
        )

    def get_response(self, input_text: str, **model_kwargs) -> str:
        input_ids = self.tokenizer(input_text, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**input_ids, **model_kwargs)
        return self.tokenizer.decode(outputs[0])


model = Gemma()
