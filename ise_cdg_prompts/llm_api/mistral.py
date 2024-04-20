import torch
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from langchain import PromptTemplate, LLMChain, HuggingFacePipeline

from .main import LLM_API


class MistralBase(LLM_API):
    model_path: str

    def __init__(self) -> None:
        super().__init__()
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_4bit = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            quantization_config=quantization_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        pipeline_inst = pipeline(
            "text-generation",
            model=model_4bit,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_length=2500,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        self.pipeline = HuggingFacePipeline(pipeline=pipeline_inst)

    def get_response(self, prompt: str) -> str:
        template = """[INST]{inquiry}[/INST]"""
        prompt_template = PromptTemplate(template=template, input_variables=["inquiry"])
        return LLMChain(prompt=prompt_template, llm=self.pipeline).run(
            {"inquiry": prompt}
        )


class Mistral(MistralBase):
    name: str = "Mistral"
    model_path: str = "mistralai/Mistral-7B-Instruct-v0.1"


class Mixtral(MistralBase):
    name: str = "Mixtral"
    model_path: str = "mistralai/Mixtral-8x7B-v0.1"
