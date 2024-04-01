from typing import TYPE_CHECKING, List
from ise_cdg_prompts.prompt_generation_visitor.main import PromptGenerationVisitor
from ise_cdg_prompts.utils.pipeline import Pipeline

if TYPE_CHECKING:
    from ise_cdg_prompts.dataset import CodeMarkdown
    from ise_cdg_prompts.task import Task


class AshkanPromptGenerator(PromptGenerationVisitor):
    def visit_task(self, task: "Task") -> str:
        return (
            "You are an expert Python programmer, please describe the functionality of the method:\n"
            + self.visit_templates(task.templates)
            + f"\n#Code\n{task.question.code}\n#Summary:"
        )

    def visit_templates(self, templates: List["CodeMarkdown"]) -> str:
        return "".join(
            [
                self.visit_template(template, index)
                for index, template in enumerate(templates)
            ]
        )

    def visit_template(self, template: "CodeMarkdown", index: int) -> str:
        # TODO bug: reverse code-markdown
        return f"#Code\n{template.markdown}\n" + f"#Summary: {template.code}\n"