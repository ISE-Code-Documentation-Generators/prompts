import random
from typing import TYPE_CHECKING, List
from tqdm import tqdm

from ise_cdg_prompts.dataset import CodeMarkdown
from ise_cdg_prompts.sample.main import TaskSampler
from ise_cdg_prompts.task import Task

if TYPE_CHECKING:
    from ise_cdg_data.dataset import DatasetFilterByIndexRaw


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