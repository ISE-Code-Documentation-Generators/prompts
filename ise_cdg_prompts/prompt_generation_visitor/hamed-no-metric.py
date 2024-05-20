from typing import TYPE_CHECKING, List
from ise_cdg_prompts.prompt_generation_visitor.main import PromptGenerationVisitor

if TYPE_CHECKING:
    from ise_cdg_prompts.dataset import CodeMarkdown, CodeMarkdownMetrics
    from ise_cdg_prompts.task import TaskMetrics


class HamedNoMetricPromptGenerator(PromptGenerationVisitor):
    def visit_task(self, task: "TaskMetrics") -> str:
        return (
            "Suppose you are an expert Python programmer. Look at these methods and their summaries:\n"
            + self.visit_templates(task.templates)
            + "\nNow given this code, please give me the summary of the code:\n"
            + f"\n#Code:\n{task.question.code}"
        )

    def visit_templates(self, templates: List["CodeMarkdown"]) -> str:
        return "".join(
            [
                self.visit_template(template, index)
                for index, template in enumerate(templates)
            ]
        )

    def visit_template(self, template: "CodeMarkdownMetrics", index: int) -> str:
        code_prompt = f"#Code:\n{template.code}\n"
        summary_prompt = f"#Summary: {template.markdown}\n"
        return code_prompt + summary_prompt
