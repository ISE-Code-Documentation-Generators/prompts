# !pip install -q -U langchain transformers bitsandbytes accelerate

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import PromptTemplate, LLMChain, HuggingFacePipeline
import torch

from .main import LLM_API


class Falcon(LLM_API):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        model_path = "Rocketknight1/falcon-rw-1b"  
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_4bit = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto",
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)        
        self.__pipe = pipeline(
            "text-generation",
            model=model_4bit,
            tokenizer=tokenizer,
            use_cache=True,
            device_map="auto",
            max_length=2500,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
    )
        
    def __prompt_model(self, inp):
        template = """[INST]{inquiry}[/INST]"""
        llm = HuggingFacePipeline(pipeline=self.__pipe)
        prompt_template = PromptTemplate(template=template, input_variables=["inquiry"])
        return LLMChain(prompt=prompt_template, llm=llm).run({"inquiry":inp})
    
    def get_response(self, prompt: str) -> str:
        return str(self.__prompt_model(prompt))
    
    
