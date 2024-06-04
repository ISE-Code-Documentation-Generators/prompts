from typing import TYPE_CHECKING, List
from ise_cdg_prompts.prompt_generation_visitor.main import PromptGenerationVisitor

if TYPE_CHECKING:
    from ise_cdg_prompts.dataset import CodeMarkdown, CodeMarkdownMetrics
    from ise_cdg_prompts.task import TaskMetrics


class AlirezaWithMetricPromptGenerator(PromptGenerationVisitor):
    def visit_task(self, task: "TaskMetrics") -> str:
        return (
            "Give me the summary of the code from the corresponding code given its code metrics. Codes are delimited by triple backquotes. Summaries and code metrics are delimited by triple quotes:\n"
            + self.visit_templates(task.templates)
            + f'Code {len(task.templates) + 1}: ```{task.question.code}```\nCode Metrics {len(task.templates) + 1}: """{task.question.metrics_string}"""\nSummary {len(task.templates) + 1}:'
        )

    def visit_templates(self, templates: List["CodeMarkdown"]) -> str:
        return "".join(
            [
                self.visit_template(template, index)
                for index, template in enumerate(templates)
            ]
        )

    def visit_template(self, template: "CodeMarkdownMetrics", index: int) -> str:
        code_prompt = f"Code {index+1}: ```{template.code}```\n"
        code_metrics_prompt = f'Code Metrics {index+1}: """{template.metrics_string}"""\n'
        summary_prompt = f'Summary {index+1}: """{template.markdown}"""\n'
        return code_prompt + code_metrics_prompt + summary_prompt
