from typing import Dict, List
from ise_cdg_prompts.utils.pipeline import Pipeline


import unittest


class KossherLLMTest2(unittest.TestCase):
    def setUp(self):
        from ise_cdg_prompts.utils.custom_io import JSON_IO

        self.io = JSON_IO("./")

    def file_name_test_default(self) -> str:
        return "gemma_quantized_llm_results.json"

    def test_default(self):
        # from ise_cdg_prompts.llm_api.gemma import model

        # kossher: List[str] = (
        #     Pipeline(gemma_llm_logs["inputs"])
        #     .to_map(lambda model_input: model.get_response(**model_input))
        #     .to_list()
        # )
        # self.io.write(
        #     {"inputs": [], "outputs": kossher2}, self.file_name_test_default()
        # )
        gemma_llm_logs: Dict = self.io.read(self.file_name_test_default())
        # print(kossher2[0])
        # print("-------")
        # print(gemma_llm_logs["outputs"][0])
        # self.assertEqual(kossher2, gemma_llm_logs["outputs"])
        self.assertEqual(len(kossher2), len(gemma_llm_logs["outputs"]))
        Pipeline(kossher2).to_map(self.assertIsNotNone)


unittest.main(argv=[""], defaultTest=KossherLLMTest2.__name__, verbosity=2, exit=False)