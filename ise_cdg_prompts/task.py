from typing import TYPE_CHECKING, List


if TYPE_CHECKING:
    from ise_cdg_prompts.dataset import CodeMarkdown

from ise_cdg_prompts.prompt_generation_visitor import TaskPromptGenerationVisitor


class Task:
    def __init__(
        self,
        question: "CodeMarkdown",
        templates: List["CodeMarkdown"],
    ):
        self.templates = templates
        self.question = question
        self.__visitor = TaskPromptGenerationVisitor()

    def get_prompt(self) -> str:
        return self.__visitor.visit_task(self)

    def get_ground_truth(self) -> str:
        return self.question.markdown

    def to_json(self):
        return [self.get_prompt(), self.get_ground_truth()]
