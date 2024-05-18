from typing import TYPE_CHECKING, List
from ise_cdg_prompts.prompt_generation_visitor.main import PromptGenerationVisitor
from ise_cdg_prompts.utils.pipeline import Pipeline

if TYPE_CHECKING:
    from ise_cdg_prompts.dataset import CodeMarkdown, CodeMarkdownMetrics
    from ise_cdg_prompts.task import TaskMetrics


class HamedPromptGenerator(PromptGenerationVisitor):
    def visit_task(self, task: "TaskMetrics") -> str:
        code_metrics_prompt = ", ".join(task.question.metrics)
        return (
            "You are an expert Python programmer, please describe the functionality of the method:\n"
            + self.visit_templates(task.templates)
            + f"\n#Code\n{task.question.code}\n#Code Metrics: {code_metrics_prompt}\n#Summary:"
        )

    def visit_templates(self, templates: List["CodeMarkdown"]) -> str:
        return "".join(
            [
                self.visit_template(template, index)
                for index, template in enumerate(templates)
            ]
        )

    def visit_template(self, template: "CodeMarkdownMetrics", index: int) -> str:
        code_prompt = f"#Code\n{template.code}\n" 
        code_metrics_prompt = f"#Code Metrics\n{", ".join(template.metrics)}\n"
        summary_prompt = f"#Summary: {template.markdown}\n"
        return code_prompt + code_metrics_prompt + summary_prompt
