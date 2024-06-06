from typing import TYPE_CHECKING, List
from tqdm import tqdm

from ise_cdg_prompts.dataset import CodeMarkdown, CodeMarkdownMetrics
from ise_cdg_prompts.sample.main import TaskSampler
from ise_cdg_prompts.task import Task, TaskMetrics

if TYPE_CHECKING:
    from ise_cdg_data.dataset import DatasetFilterByIndexRaw


# Zero shot without metrics in prompt
class ZeroShotTaskSampler(TaskSampler):
    def __init__(
        self,
        test_dataset: "DatasetFilterByIndexRaw",
    ) -> None:
        super().__init__()
        self.test_dataset = test_dataset

    def generate_samples(self) -> List[Task]:
        tasks: List[Task] = []
        for index in tqdm(range(len(self.test_dataset))):
            query_code, query_markdown = (
                self.test_dataset.get_raw_item("source", index),
                self.test_dataset.get_raw_item("markdown_text", index),
            )
            question = CodeMarkdown(query_code, query_markdown)
            templates: List["CodeMarkdown"] = []
            tasks.append(Task(question, templates))
        return tasks


# Zero shot with metrics in prompt
class ZeroShotTaskSamplerMetrics(TaskSampler):
    def __init__(
        self,
        test_dataset: "DatasetFilterByIndexRaw",
    ) -> None:
        super().__init__()
        self.test_dataset = test_dataset

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
            tasks.append(TaskMetrics(question, templates))
        return tasks
