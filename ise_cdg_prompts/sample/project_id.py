import random
from typing import TYPE_CHECKING, List
from ise_cdg_prompts.sample.main import TaskSampler

if TYPE_CHECKING:
    from ise_cdg_prompts.dataset import SimplePromptDataset
    from ise_cdg_prompts.task import Task


class ProjectIDTaskSampler(TaskSampler):
    def __init__(
        self,
        dataset: "SimplePromptDataset",
        sample_size: int,
        shot_size: int,
    ) -> None:
        super().__init__(dataset)
        self.sample_size = sample_size
        self.shot_size = shot_size

    def generate_samples(self) -> List["Task"]:
        # TODO breaking PLK
        df = self.dataset.df
        samples = []
        for pid in df["project_ID"].unique():
            qid = qid = random.sample(range(0, len(df[df["project_ID"] == pid])), 1)[0]
            template_indices = [
                ind
                for ind in random.sample(
                    range(0, self.dataset.__len__()),
                    min(self.shot_size, len(df[df["project_ID"] == pid])),
                )
                if ind != qid
            ]
            task = self._indices_to_task(
                question_index=qid,
                template_indices=template_indices,
            )
            samples.append(task)
        return samples
