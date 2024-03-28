from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, List

from ise_cdg_prompts.dataset import CodeMarkdown
from ise_cdg_prompts.utils.pipeline import Pipeline


from ise_cdg_prompts.task import Task

if TYPE_CHECKING:
    from ise_cdg_prompts.dataset import PromptDataset


class TaskSampler(metaclass=ABCMeta):
    def __init__(self, dataset: "PromptDataset") -> None:
        self.dataset = dataset

    @abstractmethod
    def generate_samples(self) -> List["Task"]:
        pass

    def _indices_to_task(
        self,
        question_index: int,
        template_indices: List["int"],
    ) -> Task:
        return Task(
            question=self.dataset[question_index],
            templates=Pipeline(template_indices)
            .to_map(self.dataset.__getitem__)
            .to_list(),
        )
