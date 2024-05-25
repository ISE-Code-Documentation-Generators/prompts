from typing import TYPE_CHECKING, List
from ise_cdg_prompts.prompt_generation_visitor.main import PromptGenerationVisitor

if TYPE_CHECKING:
    from ise_cdg_prompts.dataset import CodeMarkdown, CodeMarkdownMetrics
    from ise_cdg_prompts.task import TaskMetrics


class ZeroShotPromptGenerator(PromptGenerationVisitor):
    def visit_task(self, task: "TaskMetrics") -> str:
        sample_code = "def _combine_transfers(self, result):\ntransfers = {}\nfor reaction_id, c1, c2, form in result:\nkey = reaction_id, c1, c2\ncombined_form = transfers.setdefault(key, Formula())\ntransfers[key] = combined_form | form\n\nfor (reaction_id, c1, c2), form in iteritems(transfers):\nyield reaction_id, c1, c2, form"
        sample_summary = "Combine multiple pair transfers into one."
        return (
            # "Suppose you are an expert Python programmer."
            # + "\nLook at this python code and its summary:\n"
            # + f"\n#Code:\n{sample_code}\n#Summary:\n{sample_summary}\n"
            # + "\nNow given this code, please give me the summary of the code\n"
            "Suppose you are an expert Python programmer. Give me the summary of the code from the corresponding code:\n"
            + f"\nCode 1: ```\n{task.question.code}\n```\nSummary 1:"
        )
