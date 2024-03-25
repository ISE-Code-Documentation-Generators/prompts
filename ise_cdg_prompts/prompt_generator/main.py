import pandas as pd

# Load dataset containing markdown and code cells
dataset_path = "final_dataset.csv"
dataset = pd.read_csv(dataset_path)


def dataset_get_markdown(index: int):
    return str(dataset.iloc[index]["markdown"])


def dataset_get_source(index: int):
    return str(dataset.iloc[index]["source"])

def dataset_len():
    return dataset.shape[0]


from ise_cdg_prompts.utils.pipeline import Pipeline


class Template:
    def __init__(self, index: int, row_index: int):
        self.index = index
        self.row_index = row_index

    def representative_index(self):
        return str(self.index + 1)

    def generate_prompt(self):
        return (
            f"Start Markdown {self.representative_index()}: {dataset_get_markdown(self.row_index)}\n"
            + f"Start Code {self.representative_index()}: {dataset_get_source(self.row_index)}\n"
        )


class Sample:
    def __init__(self, template_indices, question_index):
        self.templates = (
            Pipeline(range(len(template_indices)))
            .to_map(
                lambda template_index: Template(
                    template_index, template_indices[template_index]
                )
            )
            .to_list()
        )
        self.question_index = question_index

    def generate_prompt(self):
        return (
            "For now, Just read these template Markdown and Code pairs. \n"
            + "".join(
                Pipeline(self.templates)
                .to_map(lambda template: template.generate_prompt())
                .to_list()
            )
            + f"\n Then, Generate markdown for the below code according to the pairs.\n Code: {dataset_get_source(self.question_index)}"
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
