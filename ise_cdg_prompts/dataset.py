from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple, List
import typing
import pandas as pd
from ise_cdg_data.dataset import Md4DefDatasetInterface
from torch._tensor import Tensor


@dataclass
class CodeMarkdown:
    code: str
    markdown: str


@dataclass
class CodeMarkdownMetrics:
    code: str
    markdown: str
    metrics: List[str]
    metrics_string: str
    #TODO

class PromptDataset(Md4DefDatasetInterface):
    @abstractmethod
    def __getitem__(self, index) -> "CodeMarkdown":
        pass


class SimplePromptDataset(PromptDataset):
    default_md_key = "markdown"
    default_src_key = "source"

    def __init__(
        self,
        path: str,
        md_key: typing.Optional[str] = None,
        src_key: typing.Optional[str] = None,
    ) -> None:
        super().__init__()
        self.df: "pd.DataFrame" = pd.read_csv(path)
        self.md_key = md_key or self.default_md_key
        self.src_key = src_key or self.default_src_key
        self.__remove_null_values(df=self.df)

    def __remove_null_values(self, df):
        df.dropna(subset=[self.src_key, self.md_key], inplace=True)

    def __get_markdown(self, index: int) -> str:
        return str(self.df.iloc[index][self.md_key])

    def __get_source(self, index: int) -> str:
        return str(self.df.iloc[index][self.src_key])

    def __getitem__(self, index) -> "CodeMarkdown":
        return CodeMarkdown(
            code=self.__get_source(index),
            markdown=self.__get_markdown(index),
        )

    def __len__(self) -> int:
        return self.df.shape[0]
