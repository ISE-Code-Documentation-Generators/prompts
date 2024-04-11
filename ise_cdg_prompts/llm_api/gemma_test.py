from typing import Dict, List
from ise_cdg_prompts.utils.pipeline import Pipeline


import unittest


class KossherLLMTest(unittest.TestCase):
    def setUp(self):
        from ise_cdg_prompts.utils.custom_io import JSON_IO

        self.io = JSON_IO("./")

    def file_name_test_default(self) -> str:
        return "gemma_llm_results.json"

    def test_default(self):
        from ise_cdg_prompts.llm_api.gemma import model

        gemma_llm_logs: Dict = self.io.read(self.file_name_test_default())
        kossher: List[str] = (
            Pipeline(gemma_llm_logs["inputs"])
            .to_map(lambda model_input: model.get_response(**model_input))
            .to_list()
        )
        # self.io.write(
        #     {"inputs": model_inputs, "outputs": kossher}, self.file_name_test_default()
        # )
        self.assertEqual(kossher, gemma_llm_logs["outputs"])


unittest.main(argv=[""], defaultTest="KossherLLMTest", verbosity=2, exit=False)
