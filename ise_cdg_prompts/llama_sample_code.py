from typing import List

from ise_cdg_data.dataset import Md4DefDatasetInterface
from ise_cdg_prompts.prompt_generator import Task, PromptDataset, TaskGenerator
from ise_cdg_prompts.sample import PromptSampler
from ise_cdg_prompts.sample.main import PromptSample
from ise_cdg_prompts.sample.random import RandomPromptSampler
from ise_cdg_prompts.utils.pipeline import Pipeline


sample_size = 10
shot_size = 4
dataset = PromptDataset(path="final_dataset.csv")
prompt_sampler = RandomPromptSampler(
    shot_size=shot_size,
    sample_size=sample_size,
)
tasks = TaskGenerator(
    dataset=dataset,
    prompt_sampler=prompt_sampler,
)()


print(tasks[0].get_prompt(), tasks[0].get_ground_truth())


from ise_cdg_prompts.llm_api import LLM_API
from ise_cdg_prompts.llm_api.llama import Llama_API

llm_api: "LLM_API" = Llama_API()

task_responses = Pipeline(tasks).to_map(
    lambda task: llm_api.get_response(task.get_prompt())
)

task_responses.to_map(print)
