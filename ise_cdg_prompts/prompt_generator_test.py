import unittest

from ise_cdg_prompts.custom_io import JSON_IO


class PromptGeneratorTest(unittest.TestCase):
    def test_default(self):
        from prompt_generator import generate_prompt_data, Sample, dataset

        shot_step = 4
        json_io = JSON_IO("ise_cdg_prompts")
        self.assertEqual(
            json_io.read('prompt_generation.json'),
            list(generate_prompt_data(
                samples=[
                    Sample(
                        template_indices=range(0, dataset.shape[0], shot_step),
                        question_index=dataset.shape[0] // 2,
                    )
                ]
            ))
        )

if __name__ == "__main__":
    unittest.main()