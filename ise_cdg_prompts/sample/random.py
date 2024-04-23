import random
from typing import TYPE_CHECKING, List

from ise_cdg_prompts.sample.main import TaskSampler
from ise_cdg_prompts.task import Task
from ise_cdg_prompts.utils.pipeline import Pipeline

if TYPE_CHECKING:
    from ise_cdg_prompts.dataset import PromptDataset


class RandomTaskSampler(TaskSampler):
    def __init__(
        self,
        dataset: "PromptDataset",
        shot_size: int,
        sample_size: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.__shot_size = shot_size
        self.__sample_size = sample_size

    def generate_samples(self) -> List[Task]:
        return [
            self._indices_to_task(
                dataset=self.dataset,
                question_index=self.__generate_question_index(),
                template_indices=self.__generate_template_indices(),
            )
            for i in range(self.__sample_size)
        ]

    def __generate_question_index(self) -> int:
        return random.sample(range(0, self.dataset.__len__()), 1)[0]

    def __generate_template_indices(self) -> List[int]:
        return random.sample(range(0, self.dataset.__len__()), self.__shot_size)
