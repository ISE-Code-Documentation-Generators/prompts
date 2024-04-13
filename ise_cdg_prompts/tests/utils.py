from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from unittest import TestCase
    from ise_cdg_prompts.task import Task
    from ise_cdg_prompts.prompt_generation_visitor.main import PromptGenerationVisitor


from ise_cdg_prompts.utils.pipeline import Pipeline


class AssertionUtils:
    def __init__(self) -> None:
        from ise_cdg_prompts.utils.custom_io import JSON_IO

        self.io = JSON_IO("ise_cdg_prompts/tests")

    def assert_tasks_validity_with_prompt(
        self,
        unittest_module: "TestCase",
        tasks: List["Task"],
        prompt_generation_visitor: "PromptGenerationVisitor",
        expected_tasks_file_name: str,
    ):
        test_output = (
            Pipeline(tasks)
            .to_map(lambda task: task.to_json(prompt_generation_visitor))
            .to_list()
        )
        # self.io.write(test_output, expected_tasks_file_name)
        expected_output = self.io.read(expected_tasks_file_name)
        unittest_module.assertEqual(test_output, expected_output)

    def assert_tasks_validity_with_prompt_and_ground_truth(
        self,
        unittest_module: "TestCase",
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
        unittest_module.assertEqual(output, expected_output)
