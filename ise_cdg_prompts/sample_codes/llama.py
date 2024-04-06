from typing import TYPE_CHECKING
from ise_cdg_prompts.utils.pipeline import Pipeline


from ise_cdg_prompts.dataset import SimplePromptDataset
from ise_cdg_prompts.sample.random import RandomTaskSampler

prompt_sampler = RandomTaskSampler(
    dataset=SimplePromptDataset(path="final_dataset.csv"),
    sample_size=10,
    shot_size=4,
)

if TYPE_CHECKING:
    from ise_cdg_prompts.llm_api import LLM_API
    from ise_cdg_prompts.utils.custom_io import Custom_IO
    from ise_cdg_prompts.prompt_generation_visitor.main import PromptGenerationVisitor

from ise_cdg_prompts.utils.custom_io import JSON_IO
from ise_cdg_prompts.llm_api.llama import Llama_API
from ise_cdg_prompts.prompt_generation_visitor.ashkan import AshkanPromptGenerator

llm_api: "LLM_API" = Llama_API()
io: "Custom_IO" = JSON_IO(".")
prompt_generation_visitor: "PromptGenerationVisitor" = AshkanPromptGenerator()

tasks = prompt_sampler.generate_samples()
sample_outputs = (
    Pipeline(tasks)
    .to_map(
        lambda task: {
            "prompt": task.get_prompt(visitor=prompt_generation_visitor),
            "response": llm_api.get_response(
                task.get_prompt(visitor=prompt_generation_visitor)
            ),
            "ground_truth": task.get_ground_truth(),
        }
    )
    .to_list()
)

io.write(sample_outputs, "samples.json")
