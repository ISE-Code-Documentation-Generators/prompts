import random
from typing import TYPE_CHECKING, List

from ise_cdg_prompts.dataset import CodeMarkdown
from ise_cdg_prompts.sample.main import TaskSampler
from ise_cdg_prompts.task import Task
from ise_cdg_data.information_retrieval import CodeRetrieval, load_semantical_ir_model

if TYPE_CHECKING:
    from ise_cdg_data.dataset import DatasetFilterByIndexRaw


class IRBasedTaskSampler(TaskSampler):
    def __init__(
        self,
        train_dataset: "DatasetFilterByIndexRaw",
        test_dataset: "DatasetFilterByIndexRaw",
        shot_size: int,
        device,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.__shot_size = shot_size
        self.ir: "CodeRetrieval" = load_semantical_ir_model(
            [
                self.train_dataset.get_raw_item("source", i)
                for i in range(len(self.train_dataset))
            ],
            device,
            [
                self.train_dataset.get_raw_item("embeding", i)
                for i in range(len(self.train_dataset))
            ],
        )

    def generate_samples(self) -> List[Task]:
        tasks: List[Task] = []
        for index in range(len(self.test_dataset)):
            query_code, query_markdown = (
                self.test_dataset.get_raw_item("source", index),
                self.test_dataset.get_raw_item("markdown", index),
            )
            results = self.ir.get_similar(query_code)[: self.__shot_size]
            question = CodeMarkdown(query_code, query_markdown)
            templates: List["CodeMarkdown"] = []
            for result in results:
                ind = result["corpus_id"]
                template_code, template_markdown = (
                    self.train_dataset.get_raw_item("source", ind),
                    self.train_dataset.get_raw_item("markdown", ind),
                )
                templates.append(CodeMarkdown(template_code, template_markdown))
            tasks.append(Task(question, templates))
        return tasks
