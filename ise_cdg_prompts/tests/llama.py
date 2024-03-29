from typing import List
import unittest

from ise_cdg_prompts.tests.main import PromptsUnitTest


class LlamaUnitTests(PromptsUnitTest):
    def test_sampler(self):
        import random
        from ise_cdg_prompts.dataset import SimplePromptDataset
        from ise_cdg_prompts.sample.random import RandomTaskSampler

        random.seed(0)
        prompt_sampler = RandomTaskSampler(
            dataset=SimplePromptDataset(path="final_dataset.csv"),
            sample_size=10,
            shot_size=4,
        )

        self._assert_tasks_validity(
            tasks=prompt_sampler.generate_samples(),
            expected_tasks_file_name="llama_results.json",
        )


if __name__ == "__main__":
    unittest.main()
