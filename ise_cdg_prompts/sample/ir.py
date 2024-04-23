import random
from typing import TYPE_CHECKING, List

from ise_cdg_prompts.dataset import CodeMarkdown
from ise_cdg_prompts.sample.main import TaskSampler
from ise_cdg_prompts.task import Task
from ise_cdg_data.information_retrieval import CodeRetrieval, load_semantical_ir_model

if TYPE_CHECKING:
    from ise_cdg_data.dataset import Md4DefDatasetInterface


class IRBasedTaskSampler(TaskSampler):
    def __init__(
        self,
        train_dataset: "Md4DefDatasetInterface",
        test_dataset: "Md4DefDatasetInterface",
        shot_size: int,
        device,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.__shot_size = shot_size
        self.ir: "CodeRetrieval" = load_semantical_ir_model(
            self.train_dataset.source.to_list(),
            device,
            self.train_dataset.df["embeding"].to_list(),
        )

    def generate_samples(self) -> List[Task]:
        tasks: List[Task] = []
        for index in range(len(self.test_dataset.source)):
            query_code, query_markdown = (
                self.test_dataset.source.iloc[index],
                self.test_dataset.md.iloc[index],
            )
            results = self.ir.get_similar(query_code)[: self.__shot_size]
            question = CodeMarkdown(query_code, query_markdown)
            templates: List["CodeMarkdown"] = []
            for result in results:
                ind = result["corpus_id"]
                template_code, template_markdown = (
                    self.train_dataset.source.iloc[ind],
                    self.train_dataset.markdown.iloc[ind],
                )
                templates.append(CodeMarkdown(template_code, template_markdown))
            tasks.append(Task(question, templates))
        return tasks
