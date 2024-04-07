import unittest


class KossherTest(unittest.TestCase):
    def setUp(self):
        from ise_cdg_prompts.utils.custom_io import JSON_IO

        self.io = JSON_IO("./")

    def file_name_test_default(self) -> str:
        return "gemma_results.json"

    def test_default(self):
        from ise_cdg_prompts.sample_codes.gemma import generate_response, prompt_list

        # f.write(outputs[0]["generated_text"])
        kir = prompt_list[9]
        # generate_response(kir)
        # self.io.write({"jj": kir}, self.file_name_test_default())
        kos = self.io.read(self.file_name_test_default())["jj"]
        # print("kiiiir")
        # print(kir)
        # print("koooos")
        # print(kos)
        self.assertEqual(kir, kos)


unittest.main(argv=[""], defaultTest="KossherTest", verbosity=2, exit=False)
