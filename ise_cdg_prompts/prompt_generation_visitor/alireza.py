from typing import TYPE_CHECKING, List
from ise_cdg_prompts.prompt_generation_visitor.main import PromptGenerationVisitor


if TYPE_CHECKING:
    from ise_cdg_prompts.task import Task
    from ise_cdg_prompts.dataset import CodeMarkdown


class AlirezaPromptGenerationVisitor(PromptGenerationVisitor):
    def visit_task(self, task: "Task") -> str:
        return (
            self.__visit_templates(templates=task.templates)
            + "\nGenerate markdown for the bottom code according to the four samples above\n Code: "
            + task.get_ground_truth()
        )

    def __visit_template(self, code_markdown: "CodeMarkdown", index):
        result = "Start Markdown " + str(index) + ": " + code_markdown.markdown + "\n"
        result = result + "Start Code " + str(index) + ": " + code_markdown.code + "\n"
        return result

    def __visit_templates(self, templates: List[CodeMarkdown]):
        prompt = ""
        for index, template in enumerate(templates):
            prompt = prompt + self.__visit_template(
                code_markdown=template,
                index=index + 1,
            )
        return prompt