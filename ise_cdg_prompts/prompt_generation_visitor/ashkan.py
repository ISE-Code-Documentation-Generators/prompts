from typing import TYPE_CHECKING, List
from ise_cdg_prompts.prompt_generation_visitor.main import PromptGenerationVisitor
from ise_cdg_prompts.utils.pipeline import Pipeline

if TYPE_CHECKING:
    from ise_cdg_prompts.dataset import CodeMarkdown
    from ise_cdg_prompts.task import Task


class AshkanPromptGenerator(PromptGenerationVisitor):
    def visit_task(self, task: "Task") -> str:
        sentences = [
            "Hey falcon.",
            "Your going to write documention for a python code snippet.",
        ]

        shots_count = len(task.templates)
        if shots_count:
            if shots_count > 1:
                sentences += [
                    f"Initially, I've prepared for you {shots_count} examples each consisting of a code snippet and its corresponding documentaion.",
                    "Read them and learn how to write good documentation.\n",
                    "After these examples the query code is given and its summary is empty which you are going to fill.\n",
                ]
            else:
                sentences += [
                    f"Initially, I've prepared for you one example consisting of a code snippet and its corresponding documentaion.",
                    "Read it and learn how to write good documentation.\n",
                    "After this example the query code is given and its summary is empty which you are going to fill.\n",
                ]
        
        return (
            "".join(sentences)
            + self.visit_templates(task.templates)
            + f"Query:\nCode:\n{task.question.code}\nSummary:\n"
        )

    def visit_templates(self, templates: List["CodeMarkdown"]) -> str:
        return "".join(
            [
                self.visit_template(template, index)
                for index, template in enumerate(templates)
            ]
        )

    def visit_template(self, template: "CodeMarkdown", index: int) -> str:
        return f"Example {index + 1}:\nCode:\n{template.code}\nSummary:\n{template.markdown}\n"