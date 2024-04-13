from typing import TYPE_CHECKING, List
import unittest


from ise_cdg_prompts.sample import TaskSampler
from ise_cdg_prompts.tests.utils import AssertionUtils

if TYPE_CHECKING:
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


class GeneralTaskFunctionalityTest(unittest.TestCase):
    def test_default(self):
        from ise_cdg_prompts.dataset import SimplePromptDataset
        from ise_cdg_prompts.prompt_generation_visitor.sepehr import (
            SepehrPromptGenerationVisitor,
        )

        AssertionUtils().assert_tasks_validity(
            self,
            tasks=MockTaskSampler(
                dataset=SimplePromptDataset(path="final_dataset.csv")
            ).generate_samples(),
            expected_tasks_file_name="task_results.json",
            prompt_generation_visitor=SepehrPromptGenerationVisitor(),
        )


if __name__ == "__main__":
    unittest.main()
