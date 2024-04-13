from ise_cdg_prompts.tests.main import PromptsUnitTest
from ise_cdg_prompts.tests.utils import AssertionUtils

import unittest


class DollyUnitTest(unittest.TestCase):
    def test_sampler(self):
        import random
        from ise_cdg_prompts.dataset import SimplePromptDataset
        from ise_cdg_prompts.sample.project_id import ProjectIDTaskSampler
        from ise_cdg_prompts.prompt_generation_visitor.ashkan import (
            AshkanPromptGenerator,
        )

        random.seed(0)
        AssertionUtils().assert_tasks_validity(
            self,
            tasks=ProjectIDTaskSampler(
                dataset=SimplePromptDataset(path="samples_dataset.csv"),
                sample_size=10,
                shot_size=5,
            ).generate_samples(),
            expected_tasks_file_name="dolly_results.json",
            prompt_generation_visitor=AshkanPromptGenerator(),
        )


if __name__ == "__main__":
    unittest.main()

# unittest.main(argv=[""], defaultTest="DollyUnitTest", verbosity=2, exit=False)
