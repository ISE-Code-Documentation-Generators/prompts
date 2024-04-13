import unittest

from ise_cdg_prompts.tests.utils import AssertionUtils


class GemmaUnitTests(unittest.TestCase):
    def test_default(self):
        import random
        from ise_cdg_prompts.alireza_dataset import AlirezaDataset
        from ise_cdg_prompts.prompt_generation_visitor.alireza import (
            AlirezaPromptGenerationVisitor,
        )
        from ise_cdg_prompts.sample.random import RandomTaskSampler

        random.seed(0)
        AssertionUtils().assert_tasks_validity_with_prompt_and_ground_truth(
            self,
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
