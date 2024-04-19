# get_ipython().system("pip install pandas")
# get_ipython().system('pip install -U "transformers==4.38.1" --upgrade')
# get_ipython().system("pip install accelerate")
# get_ipython().system("pip install -i https://pypi.org/simple/ bitsandbytes")


from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ise_cdg_prompts.dataset import PromptDataset
    from ise_cdg_prompts.sample.main import TaskSampler
    from ise_cdg_prompts.prompt_generation_visitor.main import PromptGenerationVisitor

from ise_cdg_prompts.dataset import SimplePromptDataset
from ise_cdg_prompts.utils.pipeline import Pipeline


import random

random.seed(0)


from ise_cdg_prompts.prompt_generation_visitor.alireza import (
    AlirezaPromptGenerationVisitor,
)
from ise_cdg_prompts.sample.random import RandomTaskSampler

# Load dataset containing markdown and code cells
dataset: "PromptDataset" = SimplePromptDataset(path="./final_dataset.csv")
task_sampler: "TaskSampler" = RandomTaskSampler(
    dataset=dataset,
    shot_size=4,
    sample_size=10,
)
prompt_generation_visitor: "PromptGenerationVisitor" = AlirezaPromptGenerationVisitor()


tasks = task_sampler.generate_samples()
prompt_list = (
    Pipeline(tasks)
    .to_map(lambda task: prompt_generation_visitor.visit_task(task))
    .to_list()
)
grund_truth = Pipeline(tasks).to_map(lambda task: task.get_ground_truth()).to_list()


from ise_cdg_prompts.llm_api.gemma_quantized import GemmaQuantized
from ise_cdg_prompts.llm_api.gemma import Gemma

# ## Load Gemma 2b
model = Gemma()
# ## Quantized loading 7b
model = GemmaQuantized()
