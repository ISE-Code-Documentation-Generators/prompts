from ise_cdg_prompts.llm_api.main import LLM_API


class GemmaQuantized(LLM_API):
    name: str = "Gemma"
    def __init__(self) -> None:
        from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
        import torch
        import os

        os.environ["HF_TOKEN"] = "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"

        # use quantized model
        self.pipeline = pipeline(
            "text-generation",
            model="google/gemma-7b-it",
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": {"load_in_4bit": True},
            },
        )

    def get_response(self, prompt_content: str) -> str:
        messages = [
            {
                "role": "user",
                "content": prompt_content,
            },
        ]
        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipeline(
            prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )
        return outputs[0]["generated_text"]
