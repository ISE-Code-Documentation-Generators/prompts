from typing import TYPE_CHECKING, List
import unittest

from ise_cdg_data.dataset import Md4DefDatasetInterface

from ise_cdg_prompts.dataset import PromptDataset
from ise_cdg_prompts.sample import PromptSample
from ise_cdg_prompts.sample.main import PromptSampler
from ise_cdg_prompts.utils.pipeline import Pipeline

if TYPE_CHECKING:
    from ise_cdg_prompts.utils.custom_io import Custom_IO


class MockSampler(PromptSampler):
    def __question_index(self, dataset: Md4DefDatasetInterface) -> int:
        return dataset.__len__() // 2

    def __template_indices(self, dataset: Md4DefDatasetInterface) -> List[int]:
        return list(range(0, dataset.__len__(), 4))

    def generate_samples(self, dataset: Md4DefDatasetInterface) -> List[PromptSample]:
        return [
            PromptSample(
                question_index=self.__question_index(dataset),
                template_indices=self.__template_indices(dataset),
            )
        ]


class PromptGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        from ise_cdg_prompts.utils.custom_io import JSON_IO

        self.io: "Custom_IO" = JSON_IO("ise_cdg_prompts/prompt_generator")

    def __output_file_test_default(self) -> str:
        return "test_default_result.json"

    def test_default(self):
        from main import TaskGenerator

        test_input = TaskGenerator(
            dataset=PromptDataset(path="final_dataset.csv"),
            prompt_sampler=MockSampler(),
        )()
        test_output = Pipeline(test_input).to_map(lambda task: task.to_json()).to_list()

        expected_output = self.io.read(self.__output_file_test_default())
        self.assertEqual(test_output, expected_output)


if __name__ == "__main__":
    unittest.main()
