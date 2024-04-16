from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple
import typing
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
    default_md_key = "markdown"
    default_src_key = "source"

    def __init__(self, path: str, md_key: typing.Optional[str] = None, src_key: typing.Optional[str] = None) -> None:
        super().__init__()
        self.df: "pd.DataFrame" = pd.read_csv(path)
        self.__remove_null_values(df=self.df)
        self.md_key = md_key or self.default_md_key
        self.src_key = src_key or self.default_src_key
    
    @classmethod
    def __remove_null_values(cls, df):
        df.dropna(subset=["source", "markdown"], inplace=True)

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
