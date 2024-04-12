import abc
from typing import List


class LLM_API(metaclass=abc.ABCMeta):
    name: str

    @abc.abstractmethod
    def get_response(self, prompt: str) -> str:
        pass


def get_llms() -> List["LLM_API"]:
    from ise_cdg_prompts.llm_api.llama import Llama_API
    from ise_cdg_prompts.llm_api.dolly import Dolly
    from ise_cdg_prompts.llm_api.falcon import Falcon

    llms = [Dolly(), Falcon(), Llama_API()]
    return llms
