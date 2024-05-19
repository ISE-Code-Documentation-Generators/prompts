from typing import TYPE_CHECKING, List
from ise_cdg_prompts.prompt_generation_visitor.main import PromptGenerationVisitor
from ise_cdg_data.dataset.features_extractor import get_source_features_extractor
import pandas as pd


if TYPE_CHECKING:
    from ise_cdg_prompts.dataset import CodeMarkdown, CodeMarkdownMetrics
    from ise_cdg_prompts.task import TaskMetrics


class HamedPromptGenerator(PromptGenerationVisitor):
    def _get_metric_string(self, source):
        df = pd.DataFrame({'source': [source]})
        get_source_features_extractor().extract_feature_columns(code_df=df)
        df.drop(columns=['API', 'source'], inplace=True)
        metrics_map = df.to_dict(orient='records')[0]
        metrics_string = ", ".join([f"{name}: {value}" for name, value in metrics_map])
        return metrics_string

    def visit_task(self, task: "TaskMetrics") -> str:
        metrics_string = self._get_metric_string(task.question.code)
        return (
            "You are an expert Python programmer, please describe the functionality of the method:\n"
            + self.visit_templates(task.templates)
            + f"\n#Code\n{task.question.code}\n#Code Metrics\n{metrics_string}\n#Summary:"
        )

    def visit_templates(self, templates: List["CodeMarkdown"]) -> str:
        return "".join(
            [
                self.visit_template(template, index)
                for index, template in enumerate(templates)
            ]
        )

    def visit_template(self, template: "CodeMarkdownMetrics", index: int) -> str:
        code_prompt = f"#Code\n{template.code}\n"
        metrics_string = self._get_metric_string(template.code)
        code_metrics_prompt = f"#Code Metrics\n{metrics_string}\n"
        summary_prompt = f"#Summary: {template.markdown}\n"
        return code_prompt + code_metrics_prompt + summary_prompt
