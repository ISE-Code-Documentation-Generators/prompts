import unittest


class KossherTest(unittest.TestCase):
    def setUp(self):
        from ise_cdg_prompts.utils.custom_io import JSON_IO

        self.io = JSON_IO("./")

    def file_name_test_default(self) -> str:
        return "gemma_results.json"

    def test_default(self):
        from ise_cdg_prompts.sample_codes.gemma import generate_response, prompt_list, grund_truth

        # generate_response(prompt_list[9])
        jj = {"prompts": prompt_list, "ground_truths": grund_truth}
        self.io.write(jj, self.file_name_test_default())
        kos = self.io.read(self.file_name_test_default())
        # print("kiiiir")
        # print(kir)
        # print("koooos")
        # print(kos)
        self.assertEqual(jj, kos)


unittest.main(argv=[""], defaultTest="KossherTest", verbosity=2, exit=False)
