from typing import TYPE_CHECKING, List
from ise_cdg_prompts.prompt_generation_visitor.main import PromptGenerationVisitor
from ise_cdg_data.dataset.features_extractor import get_source_features_extractor
import pandas as pd


if TYPE_CHECKING:
    from ise_cdg_prompts.dataset import CodeMarkdown
    from ise_cdg_prompts.task import Task


class HamedPromptGenerator(PromptGenerationVisitor):
    def _get_metric_string(self, source):
        df = pd.DataFrame({'source': [source]})
        get_source_features_extractor().extract_feature_columns(code_df=df)
        df.drop(columns=['API', 'source'], inplace=True)
        metrics_map = df.to_dict(orient='records')[0]
        metrics_string = ", ".join([f"{name}: {value}" for name, value in metrics_map.items()])
        return metrics_string

    def visit_task(self, task: "Task") -> str:
        metrics_string = self._get_metric_string(task.question.code)
        return (
            "Suppose you are an expert Python programmer. Here is the meaning of each code metric used later: LOC means Lines of Code, BLC means Number of Blank Lines of Code, UDF means Number of User-Defined Functions, I means Number of Imports, EH means Number of error handlings, ALLC means Average Line Length of Code, NDD means Number of Visualization Data Types, NEC means Number of Executed Cells, S means Number of Statements, P means Number of Parameters, KLCID means Kind of Line of Code Identifier Density, NBD means Nested Block Depth, OPRND means Number of Operands, OPRATOR means Number of Operators, UOPRND means Number of Unique Operands, UOPRATOR means Number of Unique Operators, ID means Number of Identifiers, ALID means Average Length of Identifiers, MLID means Max Length of Identifiers, CyC means Cyclomatic Complexity, EAP means External API Popularity, LOCom means Lines of Comments, CW means Number of Comment Words.\n"
            + "Look at these methods, their code metrics, and their summaries:\n"
            + self.visit_templates(task.templates)
            + "\nNow given this code and its code metrics, please give me the summary of the code:\n"
            + f"\n#Code:\n{task.question.code}\n#Code Metrics:\n{metrics_string}"
        )

    def visit_templates(self, templates: List["CodeMarkdown"]) -> str:
        return "".join(
            [
                self.visit_template(template, index)
                for index, template in enumerate(templates)
            ]
        )

    def visit_template(self, template: "CodeMarkdown", index: int) -> str:
        code_prompt = f"#Code:\n{template.code}\n"
        metrics_string = self._get_metric_string(template.code)
        code_metrics_prompt = f"#Code Metrics:\n{metrics_string}\n"
        summary_prompt = f"#Summary: {template.markdown}\n"
        return code_prompt + code_metrics_prompt + summary_prompt
