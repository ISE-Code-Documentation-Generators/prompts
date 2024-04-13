import unittest


class KossherTest(unittest.TestCase):
    def setUp(self):
        from ise_cdg_prompts.utils.custom_io import JSON_IO

        self.io = JSON_IO("./ise_cdg_prompts/tests")

    def file_name_test_default(self) -> str:
        return "gemma_results.json"

    def test_default(self):
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            from ise_cdg_prompts.dataset import PromptDataset
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
        dataset: "PromptDataset" = AlirezaDataset(path="./final_dataset.csv")
        task_sampler: "TaskSampler" = RandomTaskSampler(
            dataset=dataset,
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
        grund_truth = Pipeline(tasks).to_map(lambda task: task.get_ground_truth()).to_list()

        # generate_response(prompt_list[9])
        jj = {"prompts": prompt_list, "ground_truths": grund_truth}
        # self.io.write(jj, self.file_name_test_default())
        kos = self.io.read(self.file_name_test_default())
        # print("kiiiir")
        # print(kir)
        # print("koooos")
        # print(kos)
        self.assertEqual(jj, kos)


unittest.main(argv=[""], defaultTest="KossherTest", verbosity=2, exit=False)
