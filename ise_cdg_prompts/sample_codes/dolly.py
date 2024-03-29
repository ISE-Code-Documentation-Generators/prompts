# pip install transformers accelerate


from typing import Dict
import os
import json

import torch, random, pandas as pd
from transformers import pipeline
from ise_cdg_utility.metrics import NLPMetricInterface, CodeMetric, get_metrics
from ise_cdg_data.tokenize import get_source_and_markdown_tokenizers

random.seed(4444)

dataset_path = "samples_dataset.csv"
dataset = pd.read_csv(dataset_path)


def dataset_get_markdown(index: int, project_id=None):
    df = dataset
    if project_id is not None:
        df = df[df["project_ID"] == project_id]
    return str(df.iloc[index]["markdown"])


def dataset_get_source(index: int, project_id=None):
    df = dataset
    if project_id is not None:
        df = df[df["project_ID"] == project_id]
    return str(df.iloc[index]["source"])


# In[ ]:


class Pipeline:
    def __init__(self, l):
        self.l = l

    def to_map(self, f):
        return Pipeline(list(map(f, self.l)))

    def to_reduce(self, f, initial=None):
        from functools import reduce

        return reduce(f, self.l, initial)

    def to_list(self):
        return self.l


class Template:
    def __init__(self, index: int, row_index: int, project_id=None):
        self.index = index
        self.row_index = row_index
        self.project_id = project_id

    def representative_index(template):
        return str(template.index + 1)

    def generate_prompt(template):
        return (
            f"#Code\n{dataset_get_markdown(template.row_index)}\n"
            + f"#Summary: {dataset_get_source(template.row_index)}\n"
        )


class Sample:
    def __init__(self, template_indices, question_index, project_id=None):
        self.templates = (
            Pipeline(range(len(template_indices)))
            .to_map(
                lambda template_index: Template(
                    template_index,
                    template_indices[template_index],
                    project_id=project_id,
                )
            )
            .to_list()
        )
        self.question_index = question_index
        self.project_id = project_id

    def generate_prompt(self):
        return (
            "You are an expert Python programmer, please describe the functionality of the method:\n"
            + "".join(
                Pipeline(self.templates)
                .to_map(lambda template: template.generate_prompt())
                .to_list()
            )
            + f"\n#Code\n{dataset_get_source(self.question_index)}\n#Summary:"
        )

    def get_ground_truth(self):
        return dataset_get_markdown(self.question_index)


def generate_prompt_data(samples):
    prompt_list = (
        Pipeline(samples).to_map(lambda sample: sample.generate_prompt()).to_list()
    )
    grund_truth = (
        Pipeline(samples).to_map(lambda sample: sample.get_ground_truth()).to_list()
    )
    return prompt_list, grund_truth


# In[ ]:


sample_size = 10
shot_size = 5


def get_samples():
    df = dataset
    samples = []
    for pid in df["project_ID"].unique():
        qid = qid = random.sample(range(0, len(df[df["project_ID"] == pid])), 1)[0]
        templates = [
            ind
            for ind in random.sample(
                range(0, dataset.shape[0]),
                min(shot_size, len(df[df["project_ID"] == pid])),
            )
            if ind != qid
        ]
        samples.append(
            Sample(
                template_indices=templates,
                question_index=qid,
            )
        )
    return samples


prompt_list, ground_truth = generate_prompt_data(get_samples())


# In[ ]:


dolly = pipeline(
    model="databricks/dolly-v2-3b",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)


# In[ ]:


def prompt_model(input):
    dolly_response = dolly(input, max_new_tokens=100)
    return dolly_response[0]["generated_text"]


# In[ ]:


metrics: Dict[CodeMetric, NLPMetricInterface] = get_metrics()
_, md_tokenizer = get_source_and_markdown_tokenizers(cleanse_markdown=False)


# In[ ]:


references = []
for i in range(len(ground_truth)):
    gt = ground_truth[i]
    reference = md_tokenizer(gt)
    references.append([reference])

print(references)


# In[ ]:


for metric in metrics.values():
    metric.set_references(references)


# In[ ]:


candidates = []
for i in range(len(prompt_list)):
    prompt = prompt_list[i]
    candidate = md_tokenizer(prompt_model(prompt))
    candidates.append(candidate)

print(candidates)


# In[ ]:


results = {}
for metric_name, metric in metrics.items():
    uncleaned_result = metric(candidates)
    result = {}
    for k, v in uncleaned_result.items():
        result[k] = float(v)
    results[metric_name.value] = result


# In[ ]:


results


# In[ ]:


# Open a file in write mode
with open("Dolly-result.json", "w") as json_file:
    # Write the JSON data to the file using json.dump()
    json.dump(results, json_file)
