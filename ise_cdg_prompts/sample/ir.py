import heapq
import random
from typing import TYPE_CHECKING, List
from tqdm import tqdm

from ise_cdg_prompts.dataset import CodeMarkdown, CodeMarkdownMetrics
from ise_cdg_prompts.sample.main import TaskSampler
from ise_cdg_prompts.task import Task, TaskMetrics
from ise_cdg_data.information_retrieval import (
    CodeRetrieval,
    load_semantical_ir_model,
    load_code_metric_ir_model,
)

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
                self.test_dataset.get_raw_item("markdown_text", index),
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
                    self.train_dataset.get_raw_item("features", ind),
                )
                templates.append(
                    CodeMarkdownMetrics(
                        template_code, template_markdown, template_features
                    )
                )
            tasks.append(TaskMetrics(question, templates))
        return tasks


class CodeMetricIRTaskSampler(TaskSampler):
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
        self.ir: "CodeRetrieval" = load_code_metric_ir_model(
            [
                self.train_dataset.get_raw_item("source", i)
                for i in range(len(train_dataset))
            ],
            device,
            [
                self.train_dataset.get_raw_item("features", i)
                for i in range(len(train_dataset))
            ],
        )

    def generate_samples(self) -> List[Task]:
        tasks: List[Task] = []
        for index in tqdm(range(len(self.test_dataset))):
            query_code, query_markdown, query_features = (
                self.test_dataset.get_raw_item("source", index),
                self.test_dataset.get_raw_item("markdown_text", index),
                self.test_dataset.get_raw_item("features", index),
            )
            results = self.ir.get_similar(query_features)[: self.__shot_size]
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


class CodeMetricIRBasedTaskSamplerMetrics(TaskSampler):
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
        self.ir: "CodeRetrieval" = load_code_metric_ir_model(
            [
                self.train_dataset.get_raw_item("source", i)
                for i in range(len(train_dataset))
            ],
            device,
            [
                self.train_dataset.get_raw_item("features", i)
                for i in range(len(train_dataset))
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
            results = self.ir.get_similar(query_features)[: self.__shot_size]
            question = CodeMarkdownMetrics(query_code, query_markdown, query_features)
            templates: List["CodeMarkdownMetrics"] = []
            for result in results:
                ind = result["corpus_id"]
                template_code, template_markdown, template_features = (
                    self.train_dataset.get_raw_item("source", ind),
                    self.train_dataset.get_raw_item("markdown_text", ind),
                    self.train_dataset.get_raw_item("features", ind),
                )
                templates.append(
                    CodeMarkdownMetrics(
                        template_code, template_markdown, template_features
                    )
                )
            tasks.append(TaskMetrics(question, templates))
        return tasks


class CodeMetricBertIRTaskSampler(TaskSampler):
    def __init__(
        self,
        train_dataset: "DatasetFilterByIndexRaw",
        test_dataset: "DatasetFilterByIndexRaw",
        shot_size: int,
        w1: float,
        device,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.__shot_size = shot_size
        self.w1 = w1
        self.w2 = 1 - w1
        self.ir_1: "CodeRetrieval" = load_semantical_ir_model(
            [
                self.train_dataset.get_raw_item("source", i)
                for i in range(len(train_dataset))
            ],
            device,
            [
                self.train_dataset.get_raw_item("embeding", i)
                for i in range(len(train_dataset))
            ],
        )

        self.ir_2: "CodeRetrieval" = load_code_metric_ir_model(
            [
                self.train_dataset.get_raw_item("source", i)
                for i in range(len(train_dataset))
            ],
            device,
            [
                self.train_dataset.get_raw_item("features", i)
                for i in range(len(train_dataset))
            ],
        )

    def generate_samples(self) -> List[Task]:
        tasks: List[Task] = []
        for index in tqdm(range(len(self.test_dataset))):
            query_code, query_markdown, query_features = (
                self.test_dataset.get_raw_item("source", index),
                self.test_dataset.get_raw_item("markdown_text", index),
                self.test_dataset.get_raw_item("features", index),
            )
            results_1 = self.ir_1.get_similar(query_code)
            results_2 = self.ir_2.get_similar(query_features)
            results_3 = [0] * len(results_1)
            for i in range(len(results_1)):
                results_3[results_1[i]["corpus_id"]] += self.w1 * results_1[i]["score"]
                results_3[results_2[i]["corpus_id"]] += self.w2 * results_2[i]["score"]
            results = heapq.nlargest(
                self.__shot_size, range(len(results_3)), key=results_3.__getitem__
            )
            question = CodeMarkdown(query_code, query_markdown)
            templates: List["CodeMarkdown"] = []
            for ind in results:
                template_code, template_markdown = (
                    self.train_dataset.get_raw_item("source", ind),
                    self.train_dataset.get_raw_item("markdown_text", ind),
                )
                templates.append(CodeMarkdown(template_code, template_markdown))
            tasks.append(Task(question, templates))
        return tasks


class CodeMetricBertIRBasedTaskSamplerMetrics(TaskSampler):
    def __init__(
        self,
        train_dataset: "DatasetFilterByIndexRaw",
        test_dataset: "DatasetFilterByIndexRaw",
        shot_size: int,
        w1: float,
        device,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.__shot_size = shot_size
        self.w1 = w1
        self.w2 = 1 - w1
        self.ir_1: "CodeRetrieval" = load_semantical_ir_model(
            [
                self.train_dataset.get_raw_item("source", i)
                for i in range(len(train_dataset))
            ],
            device,
            [
                self.train_dataset.get_raw_item("embeding", i)
                for i in range(len(train_dataset))
            ],
        )

        self.ir_2: "CodeRetrieval" = load_code_metric_ir_model(
            [
                self.train_dataset.get_raw_item("source", i)
                for i in range(len(train_dataset))
            ],
            device,
            [
                self.train_dataset.get_raw_item("features", i)
                for i in range(len(train_dataset))
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
            results_1 = self.ir_1.get_similar(query_code)
            results_2 = self.ir_2.get_similar(query_features)
            results_3 = [0] * len(results_1)
            for i in range(len(results_1)):
                results_3[results_1[i]["corpus_id"]] += self.w1 * results_1[i]["score"]
                results_3[results_2[i]["corpus_id"]] += self.w2 * results_2[i]["score"]
            results = heapq.nlargest(
                self.__shot_size, range(len(results_3)), key=results_3.__getitem__
            )
            question = CodeMarkdownMetrics(query_code, query_markdown, query_features)
            templates: List["CodeMarkdownMetrics"] = []
            for ind in results:
                template_code, template_markdown, template_features = (
                    self.train_dataset.get_raw_item("source", ind),
                    self.train_dataset.get_raw_item("markdown_text", ind),
                    self.train_dataset.get_raw_item("features", ind),
                )
                templates.append(
                    CodeMarkdownMetrics(
                        template_code, template_markdown, template_features
                    )
                )
            tasks.append(TaskMetrics(question, templates))
        return tasks
