from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple
import pandas as pd
from ise_cdg_data.dataset import Md4DefDatasetInterface
from torch._tensor import Tensor


@dataclass
class CodeMarkdown:
    code: str
    markdown: str


class PromptDataset(Md4DefDatasetInterface):
    @abstractmethod
    def __getitem__(self, index) -> "CodeMarkdown":
        pass


class SimplePromptDataset(PromptDataset):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.df: "pd.DataFrame" = pd.read_csv(path)

    def __get_markdown(self, index: int) -> str:
        return str(self.df.iloc[index]["markdown"])

    def __get_source(self, index: int) -> str:
        return str(self.df.iloc[index]["source"])

    def __getitem__(self, index) -> "CodeMarkdown":
        return CodeMarkdown(
            code=self.__get_source(index),
            markdown=self.__get_markdown(index),
        )

    def __len__(self) -> int:
        return self.df.shape[0]
