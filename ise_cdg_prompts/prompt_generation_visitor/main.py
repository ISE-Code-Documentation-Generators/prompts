import abc
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ise_cdg_prompts.task import Task


class PromptGenerationVisitor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def visit_task(self, task: "Task") -> str:
        pass
