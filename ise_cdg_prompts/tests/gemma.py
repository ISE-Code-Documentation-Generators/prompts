from typing import List
import unittest


from ise_cdg_prompts.tests.main import PromptsUnitTest


class GemmaUnitTests(PromptsUnitTest):

    def file_name_test_default(self) -> str:
        return "gemma_results.json"
    
    def test_default(self):
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            from ise_cdg_prompts.sample.main import TaskSampler
            from ise_cdg_prompts.prompt_generation_visitor.main import PromptGenerationVisitor

        from ise_cdg_prompts.utils.pipeline import Pipeline


        import random

        random.seed(0)


        from ise_cdg_prompts.alireza_dataset import AlirezaDataset
        from ise_cdg_prompts.prompt_generation_visitor.alireza import (
            AlirezaPromptGenerationVisitor,
        )
        from ise_cdg_prompts.sample.random import RandomTaskSampler

        # Load dataset containing markdown and code cells
        task_sampler: "TaskSampler" = RandomTaskSampler(
            dataset=AlirezaDataset(path="./final_dataset.csv"),
            shot_size=4,
            sample_size=10,
        )
        prompt_generation_visitor: "PromptGenerationVisitor" = AlirezaPromptGenerationVisitor()


        tasks = task_sampler.generate_samples()
        prompt_list = (
            Pipeline(tasks)
            .to_map(lambda task: prompt_generation_visitor.visit_task(task))
            .to_list()
        )
        ground_truths = Pipeline(tasks).to_map(lambda task: task.get_ground_truth()).to_list()

        output = {"prompts": prompt_list, "ground_truths": ground_truths}
        # self.io.write(output, self.file_name_test_default())
        expected_output = self.io.read(self.file_name_test_default())
        self.assertEqual(output, expected_output)


if __name__ == "__main__":
    unittest.main(argv=[""], defaultTest=GemmaUnitTests.__name__, verbosity=2, exit=False)
