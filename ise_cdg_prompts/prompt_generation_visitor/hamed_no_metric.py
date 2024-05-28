from typing import TYPE_CHECKING, List
from ise_cdg_prompts.prompt_generation_visitor.main import PromptGenerationVisitor

if TYPE_CHECKING:
    from ise_cdg_prompts.dataset import CodeMarkdown, CodeMarkdownMetrics
    from ise_cdg_prompts.task import TaskMetrics


class HamedNoMetricPromptGenerator(PromptGenerationVisitor):
    def visit_task(self, task: "TaskMetrics") -> str:
        return (
            "Suppose you are an expert Python programmer. Give me the summary of the code from the corresponding code:\n"
            + self.visit_templates(task.templates)
            + f'\nCode {len(task.templates) + 1}: ```\n{task.question.code}\n```\nSummary {len(task.templates) + 1}:'
        )

    def visit_templates(self, templates: List["CodeMarkdown"]) -> str:
        return "".join(
            [
                self.visit_template(template, index)
                for index, template in enumerate(templates)
            ]
        )

    def visit_template(self, template: "CodeMarkdownMetrics", index: int) -> str:
        code_prompt = f"Code {index+1}: ```\n{template.code}\n```\n"
        summary_prompt = f'Summary {index+1}: """\n{template.markdown}\n"""\n'
        return code_prompt + summary_prompt
