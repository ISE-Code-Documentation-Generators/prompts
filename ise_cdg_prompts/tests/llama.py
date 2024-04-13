from typing import List
import unittest

from ise_cdg_prompts.tests.utils import AssertionUtils


class LlamaUnitTests(unittest.TestCase):
    def test_sampler(self):
        from ise_cdg_prompts.prompt_generation_visitor.sepehr import (
            SepehrPromptGenerationVisitor,
        )
        from ise_cdg_prompts.dataset import SimplePromptDataset
        from ise_cdg_prompts.sample.random import RandomTaskSampler

        import random

        random.seed(0)
        prompt_sampler = RandomTaskSampler(
            dataset=SimplePromptDataset(path="final_dataset.csv"),
            sample_size=10,
            shot_size=4,
        )

        AssertionUtils().assert_tasks_validity_with_prompt(
            self,
            tasks=prompt_sampler.generate_samples(),
            expected_tasks_file_name="llama_results.json",
            prompt_generation_visitor=SepehrPromptGenerationVisitor(),
        )


if __name__ == "__main__":
    unittest.main()
