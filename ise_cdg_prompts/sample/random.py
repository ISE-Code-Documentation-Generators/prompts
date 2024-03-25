import random
from typing import List

from ise_cdg_prompts.sample.main import PromptSampler, PromptSample
from ise_cdg_data.dataset import Md4DefDatasetInterface
from ise_cdg_prompts.prompt_generator import Task, PromptDataset, TaskGenerator


class RandomPromptSampler(PromptSampler):
    def __init__(self, shot_size: int, sample_size: int) -> None:
        super().__init__()
        self.__shot_size = shot_size
        self.__sample_size = sample_size

    def generate_samples(self, dataset: "Md4DefDatasetInterface") -> List[PromptSample]:
        return [
            PromptSample(
                question_index=self.question_index(dataset),
                template_indices=self.template_indices(dataset),
            )
            for i in range(self.__sample_size)
        ]

    def question_index(self, dataset: "Md4DefDatasetInterface") -> int:
        return random.sample(range(0, dataset.__len__()), 1)[0]

    def template_indices(self, dataset: "Md4DefDatasetInterface") -> List[int]:
        return random.sample(range(0, dataset.__len__()), self.__shot_size)
