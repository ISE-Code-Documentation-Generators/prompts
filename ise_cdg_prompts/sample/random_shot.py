import random
from typing import TYPE_CHECKING, List
from tqdm import tqdm

from ise_cdg_prompts.dataset import CodeMarkdown, CodeMarkdownMetrics
from ise_cdg_prompts.sample.main import TaskSampler
from ise_cdg_prompts.task import Task, TaskMetrics

if TYPE_CHECKING:
    from ise_cdg_data.dataset import DatasetFilterByIndexRaw


# Random shot IR without metrics in prompt
class RandomShotTaskSampler(TaskSampler):
    def __init__(
        self,
        train_dataset: "DatasetFilterByIndexRaw",
        test_dataset: "DatasetFilterByIndexRaw",
        shot_size: int,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.__shot_size = shot_size

    def generate_samples(self) -> List[Task]:
        tasks: List[Task] = []
        for index in tqdm(range(len(self.test_dataset))):
            query_code, query_markdown = (
                self.test_dataset.get_raw_item("source", index),
                self.test_dataset.get_raw_item("markdown_text", index),
            )
            question = CodeMarkdown(query_code, query_markdown)
            templates: List["CodeMarkdown"] = []
            results = random.sample(range(len(self.train_dataset)), self.__shot_size)
            for ind in results:
                template_code, template_markdown = (
                    self.train_dataset.get_raw_item("source", ind),
                    self.train_dataset.get_raw_item("markdown_text", ind),
                )
                templates.append(CodeMarkdown(template_code, template_markdown))
            tasks.append(Task(question, templates))
        return tasks


# Random shot IR with metrics in prompt
class RandomShotTaskSamplerMetrics(TaskSampler):
    def __init__(
        self,
        train_dataset: "DatasetFilterByIndexRaw",
        test_dataset: "DatasetFilterByIndexRaw",
        shot_size: int,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.__shot_size = shot_size

    def generate_samples(self) -> List[TaskMetrics]:
        tasks: List[TaskMetrics] = []
        for index in tqdm(range(len(self.test_dataset))):
            query_code, query_markdown, query_features, query_features_string = (
                self.test_dataset.get_raw_item("source", index),
                self.test_dataset.get_raw_item("markdown_text", index),
                self.test_dataset.get_raw_item("features", index),
                self.test_dataset.get_raw_item("features_string", index),
            )
            question = CodeMarkdownMetrics(query_code, query_markdown, query_features, query_features_string)
            templates: List["CodeMarkdownMetrics"] = []
            results = random.sample(range(len(self.train_dataset)), self.__shot_size)
            for ind in results:
                template_code, template_markdown, template_features, template_features_string = (
                    self.train_dataset.get_raw_item("source", ind),
                    self.train_dataset.get_raw_item("markdown_text", ind),
                    self.train_dataset.get_raw_item("features", ind),
                    self.train_dataset.get_raw_item("features_string", ind),
                )
                templates.append(CodeMarkdownMetrics(template_code, template_markdown, template_features, template_features_string, ))
            tasks.append(TaskMetrics(question, templates))
        return tasks
