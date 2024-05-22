from typing import TYPE_CHECKING, List
from tqdm import tqdm

from ise_cdg_prompts.dataset import CodeMarkdown
from ise_cdg_prompts.sample.main import TaskSampler
from ise_cdg_prompts.task import Task

if TYPE_CHECKING:
    from ise_cdg_data.dataset import DatasetFilterByIndexRaw


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
