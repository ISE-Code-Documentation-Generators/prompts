from typing import TYPE_CHECKING
from ise_cdg_prompts.utils.pipeline import Pipeline


from ise_cdg_prompts.dataset import SimplePromptDataset
from ise_cdg_prompts.sample.random import RandomTaskSampler

prompt_sampler = RandomTaskSampler(
    dataset=SimplePromptDataset(path="final_dataset.csv"),
    sample_size=10,
    shot_size=4,
)

tasks = prompt_sampler.generate_samples()
print(tasks[0].get_prompt(), tasks[0].get_ground_truth())

if TYPE_CHECKING:
    from ise_cdg_prompts.llm_api import LLM_API

from ise_cdg_prompts.llm_api.llama import Llama_API

llm_api: "LLM_API" = Llama_API()

task_responses = Pipeline(tasks).to_map(
    lambda task: llm_api.get_response(task.get_prompt())
)

task_responses.to_map(print)
