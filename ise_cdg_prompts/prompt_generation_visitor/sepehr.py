from typing import TYPE_CHECKING, List

from ise_cdg_prompts.prompt_generation_visitor.main import PromptGenerationVisitor


if TYPE_CHECKING:
    from ise_cdg_prompts.task import Task
    from ise_cdg_prompts.dataset import CodeMarkdown


class SepehrPromptGenerationVisitor(PromptGenerationVisitor):
    def visit_task(self, task: "Task") -> str:
        return (
            "For now, Just read these template Markdown and Code pairs.\n"
            + f"{self.__visit_templates(task.templates)}\n"
            + f"Then, Generate markdown for the below code according to the pairs.\n"
            + f"Code: {task.question.code}\n"
        )

    def __visit_templates(self, templates: List["CodeMarkdown"]) -> str:
        return "".join(
            [
                self.__visit_template(template, index)
                for index, template in enumerate(templates)
            ]
        )

    def __visit_template(self, template: "CodeMarkdown", index: int) -> str:
        def representative_index(index: int) -> str:
            return str(index + 1)

        return (
            f"Start Markdown {representative_index(index)}: {template.markdown}\n"
            + f"Start Code {representative_index(index)}: {template.code}\n"
        )
