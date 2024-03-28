from typing import TYPE_CHECKING, List
import unittest


from ise_cdg_prompts.dataset import SimplePromptDataset
from ise_cdg_prompts.sample import TaskSampler
from ise_cdg_prompts.utils.pipeline import Pipeline

if TYPE_CHECKING:
    from ise_cdg_prompts.utils.custom_io import Custom_IO
    from ise_cdg_prompts.dataset import PromptDataset
    from ise_cdg_prompts.task import Task


class MockTaskSampler(TaskSampler):
    def generate_samples(self) -> List["Task"]:
        return [
            self._indices_to_task(
                question_index=self.__question_index(),
                template_indices=self.__template_indices(),
            )
        ]

    def __question_index(self) -> int:
        return self.dataset.__len__() // 2

    def __template_indices(self) -> List[int]:
        return list(range(0, self.dataset.__len__(), 4))


class PromptGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        from ise_cdg_prompts.utils.custom_io import JSON_IO

        self.io: "Custom_IO" = JSON_IO("ise_cdg_prompts/tests")

    def __assert_tasks_validity(
        self, tasks: List["Task"], expected_tasks_file_name: str
    ):
        test_output = Pipeline(tasks).to_map(lambda task: task.to_json()).to_list()
        expected_output = self.io.read(expected_tasks_file_name)
        self.assertEqual(test_output, expected_output)

    def test_default(self):
        self.__assert_tasks_validity(
            tasks=MockTaskSampler(
                dataset=SimplePromptDataset(path="final_dataset.csv")
            ).generate_samples(),
            expected_tasks_file_name="generator_results.json",
        )


if __name__ == "__main__":
    unittest.main()
