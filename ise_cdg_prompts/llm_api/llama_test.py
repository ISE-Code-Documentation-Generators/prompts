import unittest

from ise_cdg_prompts.llm_api.llama import Llama_API
from ise_cdg_prompts.llm_api import LLM_API

class LlamaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.llama: "LLM_API" = Llama_API()

    def test_default(self):
        test_input = "Paul Graham is "
        test_output = self.llama.get_response(test_input)
        # This can't be deterministically tested, because the output is stochastic, 
        # hence being a little different at every call.
        print(test_output)


if __name__ == "__main__":
    unittest.main()
