from ise_cdg_prompts.tests.main import PromptsUnitTest


class DollyUnitTest(PromptsUnitTest):
    def test_sampler(self):
        import random
        from ise_cdg_prompts.dataset import SimplePromptDataset
        from ise_cdg_prompts.sample.project_id import ProjectIDTaskSampler

        random.seed(0)
        self._assert_tasks_validity(
            tasks=ProjectIDTaskSampler(
                dataset=SimplePromptDataset(path="samples_dataset.csv"),
                sample_size=10,
                shot_size=5,
            ).generate_samples(),
            expected_tasks_file_name="dolly_results.json",
        )


if __name__ == "__main__":
    import unittest

    unittest.main()

# unittest.main(argv=[""], defaultTest="DollyUnitTest", verbosity=2, exit=False)
