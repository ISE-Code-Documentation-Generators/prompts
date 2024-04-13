from typing import List
import unittest


from ise_cdg_prompts.tests.main import PromptsUnitTest

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ise_cdg_prompts.task import Task
    from ise_cdg_prompts.sample.main import TaskSampler
    from ise_cdg_prompts.prompt_generation_visitor.main import PromptGenerationVisitor

from ise_cdg_prompts.utils.pipeline import Pipeline


class GemmaUnitTests(PromptsUnitTest):
    def _assert_tasks_validity(
        self,
        tasks: List["Task"],
        prompt_generation_visitor: "PromptGenerationVisitor",
        expected_tasks_file_name: str,
    ):
        prompt_list = (
            Pipeline(tasks)
            .to_map(lambda task: prompt_generation_visitor.visit_task(task))
            .to_list()
        )
        ground_truths = (
            Pipeline(tasks).to_map(lambda task: task.get_ground_truth()).to_list()
        )

        output = {"prompts": prompt_list, "ground_truths": ground_truths}
        # self.io.write(output, expected_tasks_file_name)
        expected_output = self.io.read(expected_tasks_file_name)
        self.assertEqual(output, expected_output)

    def test_default(self):
        import random

        random.seed(0)

        from ise_cdg_prompts.alireza_dataset import AlirezaDataset
        from ise_cdg_prompts.prompt_generation_visitor.alireza import (
            AlirezaPromptGenerationVisitor,
        )
        from ise_cdg_prompts.sample.random import RandomTaskSampler

        # Load dataset containing markdown and code cells
        self._assert_tasks_validity(
            tasks=RandomTaskSampler(
                dataset=AlirezaDataset(path="./final_dataset.csv"),
                shot_size=4,
                sample_size=10,
            ).generate_samples(),
            prompt_generation_visitor=AlirezaPromptGenerationVisitor(),
            expected_tasks_file_name="gemma_results.json",
        )


if __name__ == "__main__":
    unittest.main(
        argv=[""], defaultTest=GemmaUnitTests.__name__, verbosity=2, exit=False
    )
