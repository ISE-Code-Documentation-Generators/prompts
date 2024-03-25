from typing import TYPE_CHECKING
import unittest


if TYPE_CHECKING:
    from ise_cdg_prompts.utils.custom_io import Custom_IO


class PromptGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        from ise_cdg_prompts.utils.custom_io import JSON_IO

        self.io: "Custom_IO" = JSON_IO("ise_cdg_prompts/prompt_generator")

    def test_default(self):
        from main import generate_prompt_data, Sample, dataset_len

        test_input = [
            Sample(
                template_indices=range(0, dataset_len(), 4),
                question_index=dataset_len() // 2,
            )
        ]
        test_output = list(generate_prompt_data(samples=test_input))
        expected_output = self.io.read("test_default_result.json")

        self.assertEqual(test_output, expected_output)


if __name__ == "__main__":
    unittest.main()
