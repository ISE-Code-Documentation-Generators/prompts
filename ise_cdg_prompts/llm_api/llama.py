from functools import cached_property
from .main import LLM_API

class Llama_API(LLM_API):
    @classmethod
    def get_llama(cls, api_key: str):
        # pip install llama-index-program-openai llama-index-llms-llama-api llama-index
        from llama_index.llms.llama_api import LlamaAPI
        return LlamaAPI(api_key=api_key)
    
    @cached_property
    def llama(self):
        return self.get_llama(api_key = "LL-U2BMZaeGCgEdNQ56j2UyBNN8bztei5whjdqxLyMCtaMdcaSVCkm70Faq5WYF2KF4")
    
    def get_response(self, prompt: str) -> str:
        #     try:
            return self.llama.complete(prompt).text
        #     except:
        #         return "fuck"