from typing import Dict, List
from ise_cdg_prompts.utils.pipeline import Pipeline


import unittest


class GemmaQuantizedTest(unittest.TestCase):
    def setUp(self):
        from ise_cdg_prompts.utils.custom_io import JSON_IO

        self.io = JSON_IO("./")

    def file_name_test_default(self) -> str:
        return "gemma_quantized_llm_results.json"

    def test_default(self):
        from ise_cdg_prompts.sample_codes.gemma import model

        gemma_llm_logs: Dict = self.io.read(self.file_name_test_default())
        gemma_outputs = Pipeline(gemma_llm_logs["inputs"]).to_map(model.generate_response).to_list()
        self.io.write(
            {"inputs": gemma_llm_logs["inputs"], "outputs": gemma_outputs}, self.file_name_test_default()
        )
        # print(gemma_outputs[0])
        # print("-------")
        # print(gemma_llm_logs["outputs"][0])
        # self.assertEqual(gemma_outputs, gemma_llm_logs["outputs"])
        self.assertEqual(len(gemma_outputs), 2)
        Pipeline(gemma_outputs).to_map(self.assertIsNotNone)


if __name__ == "__main__":
    unittest.main(argv=[""], defaultTest=GemmaQuantizedTest.__name__, verbosity=2, exit=False)