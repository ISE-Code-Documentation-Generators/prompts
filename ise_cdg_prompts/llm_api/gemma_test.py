from typing import Dict, List
from ise_cdg_prompts.utils.pipeline import Pipeline


import unittest


class GemmaLLMTest(unittest.TestCase):
    def setUp(self):
        from ise_cdg_prompts.utils.custom_io import JSON_IO

        self.io = JSON_IO("./")

    def file_name_test_default(self) -> str:
        return "gemma_llm_results.json"

    def test_default(self):
        from ise_cdg_prompts.llm_api.gemma import model

        gemma_llm_logs: Dict = self.io.read(self.file_name_test_default())
        gemma_outputs: List[str] = (
            Pipeline(gemma_llm_logs["inputs"])
            .to_map(lambda model_input: model.get_response(**model_input))
            .to_list()
        )
        # self.io.write(
        #     {"inputs": model_inputs, "outputs": gemma_outputs}, self.file_name_test_default()
        # )
        self.assertEqual(gemma_outputs, gemma_llm_logs["outputs"])


if __name__ == "__main__":
    unittest.main()
