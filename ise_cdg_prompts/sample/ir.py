import random
from typing import TYPE_CHECKING, List
from tqdm import tqdm

from ise_cdg_prompts.dataset import CodeMarkdown, CodeMarkdownMetrics
from ise_cdg_prompts.sample.main import TaskSampler
from ise_cdg_prompts.task import Task, TaskMetrics
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
        for index in tqdm(range(len(self.test_dataset))):
            query_code, query_markdown = (
                self.test_dataset.get_raw_item("source", index),
                self.test_dataset.get_raw_item("markdown_text", index)
            )
            results = self.ir.get_similar(query_code)[: self.__shot_size]
            question = CodeMarkdown(query_code, query_markdown)
            templates: List["CodeMarkdown"] = []
            for result in results:
                ind = result["corpus_id"]
                template_code, template_markdown = (
                    self.train_dataset.get_raw_item("source", ind),
                    self.train_dataset.get_raw_item("markdown_text", ind),
                )
                templates.append(CodeMarkdown(template_code, template_markdown))
            tasks.append(Task(question, templates))
        return tasks
    
class IRBasedTaskSamplerMetrics(TaskSampler):
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

    def generate_samples(self) -> List[TaskMetrics]:
        tasks: List[TaskMetrics] = []
        for index in tqdm(range(len(self.test_dataset))):
            query_code, query_markdown, query_features = (
                self.test_dataset.get_raw_item("source", index),
                self.test_dataset.get_raw_item("markdown_text", index),
                self.test_dataset.get_raw_item("features", index),
            )
            results = self.ir.get_similar(query_code)[: self.__shot_size]
            question = CodeMarkdownMetrics(query_code, query_markdown, query_features)
            templates: List["CodeMarkdownMetrics"] = []
            for result in results:
                ind = result["corpus_id"]
                template_code, template_markdown, template_features = (
                    self.train_dataset.get_raw_item("source", ind),
                    self.train_dataset.get_raw_item("markdown_text", ind),
                    self.train_dataset.get_raw_item("features", ind)
                )
                templates.append(CodeMarkdownMetrics(template_code, template_markdown, template_features))
            tasks.append(TaskMetrics(question, templates))
        return tasks
