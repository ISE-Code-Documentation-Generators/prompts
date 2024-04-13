# get_ipython().system("pip install pandas")
# get_ipython().system('pip install -U "transformers==4.38.1" --upgrade')
# get_ipython().system("pip install accelerate")
# get_ipython().system("pip install -i https://pypi.org/simple/ bitsandbytes")


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ise_cdg_prompts.dataset import PromptDataset
    from ise_cdg_prompts.sample.main import TaskSampler
    from ise_cdg_prompts.prompt_generation_visitor.main import PromptGenerationVisitor

from ise_cdg_prompts.utils.pipeline import Pipeline


import random

random.seed(0)


from ise_cdg_prompts.alireza_dataset import AlirezaDataset
from ise_cdg_prompts.prompt_generation_visitor.alireza import (
    AlirezaPromptGenerationVisitor,
)
from ise_cdg_prompts.sample.random import RandomTaskSampler

# Load dataset containing markdown and code cells
dataset: "PromptDataset" = AlirezaDataset(path="./final_dataset.csv")
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


# ## Load Gemma 2b
from ise_cdg_prompts.llm_api.gemma_test import KossherLLMTest


# ## Quantized loading 7b
class GemmaQuantized:
    def __init__(self) -> None:
        from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
        import torch
        import os

        os.environ["HF_TOKEN"] = "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"

        # use quantized model
        self.pipeline = pipeline(
            "text-generation",
            model="google/gemma-7b-it",
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": {"load_in_4bit": True},
            },
        )

    def generate_response(self, prompt_content):
        messages = [
            {
                "role": "user",
                "content": prompt_content,
            },
        ]
        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipeline(
            prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )
        return outputs[0]["generated_text"]
    

model = GemmaQuantized()