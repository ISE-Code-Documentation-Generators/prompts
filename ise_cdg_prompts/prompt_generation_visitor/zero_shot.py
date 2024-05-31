from typing import TYPE_CHECKING, List
from ise_cdg_prompts.prompt_generation_visitor.main import PromptGenerationVisitor

if TYPE_CHECKING:
    from ise_cdg_prompts.dataset import CodeMarkdown, CodeMarkdownMetrics
    from ise_cdg_prompts.task import TaskMetrics


class ZeroShotPromptGenerator(PromptGenerationVisitor):
    def visit_task(self, task: "TaskMetrics") -> str:
        return (
            "Suppose you are an expert Python programmer. Give me the summary of the code from the corresponding code:\n"
            + f"\nCode 1: ```\n{task.question.code}\n```\nSummary 1:"
        )
