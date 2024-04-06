#  pip install transformers accelerate


from typing import Dict, List
import json

import torch, random, pandas as pd
from ise_cdg_utility.metrics import NLPMetricInterface, CodeMetric, get_metrics
from ise_cdg_data.tokenize import get_source_and_markdown_tokenizers

from ise_cdg_prompts.sample.project_id import ProjectIDTaskSampler
from ise_cdg_prompts.task import Task
from ise_cdg_prompts.utils.pipeline import Pipeline


random.seed(4444)
from ise_cdg_prompts.dataset import SimplePromptDataset


get_samples = ProjectIDTaskSampler(
    dataset=SimplePromptDataset(path="samples_dataset.csv"),
    sample_size=10,
    shot_size=5,
).generate_samples


from ise_cdg_prompts.prompt_generation_visitor.ashkan import AshkanPromptGenerator
from ise_cdg_prompts.prompt_generation_visitor.main import PromptGenerationVisitor

visitor: "PromptGenerationVisitor" = AshkanPromptGenerator()


def generate_prompt_data(samples: List["Task"]):
    prompt_list = (
        Pipeline(samples)
        .to_map(
            lambda sample: sample.get_prompt(
                visitor=visitor,
            )
        )
        .to_list()
    )
    grund_truth = (
        Pipeline(samples).to_map(lambda sample: sample.get_ground_truth()).to_list()
    )
    return prompt_list, grund_truth


def main():
    prompt_list, ground_truth = generate_prompt_data(get_samples())
    from transformers import pipeline

    dolly = pipeline(
        model="databricks/dolly-v2-3b",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    def prompt_model(input):
        dolly_response = dolly(input, max_new_tokens=100)
        return dolly_response[0]["generated_text"]

    metrics: Dict[CodeMetric, NLPMetricInterface] = get_metrics()
    _, md_tokenizer = get_source_and_markdown_tokenizers(cleanse_markdown=False)

    references = []
    for i in range(len(ground_truth)):
        gt = ground_truth[i]
        reference = md_tokenizer(gt)
        references.append([reference])

    print(references)

    for metric in metrics.values():
        metric.set_references(references)

    print("waiting for dolly")
    candidates = []
    for i in range(len(prompt_list)):
        prompt = prompt_list[i]
        candidate = md_tokenizer(prompt_model(prompt))
        candidates.append(candidate)

    print(candidates)

    #   results = {}
    #   for metric_name, metric in metrics.items():
    #       uncleaned_result = metric(candidates)
    #       result = {}
    #       for k, v in uncleaned_result.items():
    #           result[k] = float(v)
    #       results[metric_name.value] = result

    #   results

    # Open a file in write mode
    with open("Dolly-result.json", "w") as json_file:
        # Write the JSON data to the file using json.dump()
        json.dump(results, json_file)
