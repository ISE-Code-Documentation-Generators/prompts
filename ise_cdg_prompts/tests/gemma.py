import unittest


class KossherTest(unittest.TestCase):
    def setUp(self):
        from ise_cdg_prompts.utils.custom_io import JSON_IO

        self.io = JSON_IO("./")

    def test_default(self):
        from ise_cdg_prompts.sample_codes.gemma import generate_response, prompt_list

        # f.write(outputs[0]["generated_text"])
        kir = prompt_list[9]
        # generate_response(kir)
        # self.io.write({"jj": kir}, "generated_texttttt.txt")
        kos = self.io.read("generated_texttttt.txt")["jj"]
        # print("kiiiir")
        # print(kir)
        # print("koooos")
        # print(kos)
        self.assertEqual(kir, kos)


unittest.main(argv=[""], defaultTest="KossherTest", verbosity=2, exit=False)
