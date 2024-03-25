from functools import cached_property
from typing import Any, List


from ise_cdg_prompts.dataset import CodeMarkdown, PromptDataset
from ise_cdg_prompts.sample import PromptSampler, PromptSample
from ise_cdg_prompts.utils.pipeline import Pipeline


class TaskTemplateResponse:
    def __init__(
        self,
        code_markdown: "CodeMarkdown",
    ):
        self.__code_markdown = code_markdown

    def __representative_index(self, index: int) -> str:
        return str(index + 1)

    def get_prompt(self, index: int) -> str:
        return (
            f"Start Markdown {self.__representative_index(index)}: {self.__code_markdown.markdown}\n"
            + f"Start Code {self.__representative_index(index)}: {self.__code_markdown.code}\n"
        )


class Task:
    def __init__(
        self,
        code_markdown: "CodeMarkdown",
        templates: List["TaskTemplateResponse"],
    ):
        self.templates = templates
        self.__code_markdown = code_markdown

    @cached_property
    def __templates_prompt(self) -> str:
        return "".join(
            [
                template.get_prompt(index)
                for index, template in enumerate(self.templates)
            ]
        )

    def get_prompt(self) -> str:
        return (
            "For now, Just read these template Markdown and Code pairs.\n"
            + f"{self.__templates_prompt}\n"
            + f"Then, Generate markdown for the below code according to the pairs.\n"
            + f"Code: {self.__code_markdown.code}\n"
        )

    def get_ground_truth(self) -> str:
        return self.__code_markdown.markdown

    def to_json(self):
        return [self.get_prompt(), self.get_ground_truth()]


class TaskGenerator:
    def __init__(
        self,
        dataset: "PromptDataset",
        prompt_sampler: "PromptSampler",
    ) -> None:
        self.dataset = dataset
        self.prompt_sampler = prompt_sampler

    def __generate_templates(
        self, template_indices: List[int]
    ) -> List["TaskTemplateResponse"]:
        return (
            Pipeline(template_indices)
            .to_map(
                lambda template_index: TaskTemplateResponse(
                    code_markdown=self.dataset[template_index],
                )
            )
            .to_list()
        )

    def get_task(self, prompt_sample: "PromptSample") -> "Task":
        return Task(
            code_markdown=self.dataset[prompt_sample.question_index],
            templates=self.__generate_templates(prompt_sample.template_indices),
        )

    def __call__(self) -> List["Task"]:
        return (
            Pipeline(self.prompt_sampler.generate_samples(self.dataset))
            .to_map(self.get_task)
            .to_list()
        )
