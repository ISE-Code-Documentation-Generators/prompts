from abc import ABCMeta, abstractmethod
from typing import Any, List


from ise_cdg_data.dataset import Md4DefDatasetInterface


class PromptSample:
    def __init__(self, question_index: int, template_indices: List[int]) -> None:
        self.question_index = question_index
        self.template_indices = template_indices


class PromptSampler(metaclass=ABCMeta):
    @abstractmethod
    def generate_samples(self, dataset: "Md4DefDatasetInterface") -> List["PromptSample"]:
        pass
