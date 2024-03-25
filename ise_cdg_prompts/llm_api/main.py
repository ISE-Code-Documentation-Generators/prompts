import abc
class LLM_API(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_response(self, prompt: str) -> str:
        pass