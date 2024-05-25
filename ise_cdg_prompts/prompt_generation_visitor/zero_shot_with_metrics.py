from typing import TYPE_CHECKING, List
from ise_cdg_prompts.prompt_generation_visitor.main import PromptGenerationVisitor
from ise_cdg_data.dataset.features_extractor import get_source_features_extractor
import pandas as pd

if TYPE_CHECKING:
    from ise_cdg_prompts.dataset import CodeMarkdown, CodeMarkdownMetrics
    from ise_cdg_prompts.task import TaskMetrics


class ZeroShotWithMetricsPromptGenerator(PromptGenerationVisitor):
    def _get_metric_string(self, source):
        df = pd.DataFrame({'source': [source]})
        get_source_features_extractor().extract_feature_columns(code_df=df)
        df.drop(columns=['API', 'source'], inplace=True)
        metrics_map = df.to_dict(orient='records')[0]
        metrics_string = ", ".join([f"{name}: {value}" for name, value in metrics_map.items()])
        return metrics_string

    def visit_task(self, task: "TaskMetrics") -> str:
        metrics_string = self._get_metric_string(task.question.code)
        sample_code = "def _combine_transfers(self, result):\ntransfers = {}\nfor reaction_id, c1, c2, form in result:\nkey = reaction_id, c1, c2\ncombined_form = transfers.setdefault(key, Formula())\ntransfers[key] = combined_form | form\n\nfor (reaction_id, c1, c2), form in iteritems(transfers):\nyield reaction_id, c1, c2, form"
        sample_summary = "Combine multiple pair transfers into one."
        sample_metrics = "LOC: 10, BLC: 0, UDF: 1, I: 0, EH: 0, ALLC: 41.9, NDD: 0, NEC: 0, S: 8, P: 5, KLCID: 4.23, NBD: 11.6, OPRND: 14, OPRATOR: 7, UOPRND: 14, UOPRATOR: 3, ID: 38, ALID: 6, MLID: 18, CyC: 1, EAP: 0, LOCom: 0, CW: 0"
        return (
            "Suppose you are an expert Python programmer. Give me the summary of the code from the corresponding code given its code metrics:\n"
            + "(Here is the meaning of each code metric used later: LOC means Lines of Code, BLC means Number of Blank Lines of Code, UDF means Number of User-Defined Functions, I means Number of Imports, EH means Number of error handlings, ALLC means Average Line Length of Code, NDD means Number of Visualization Data Types, NEC means Number of Executed Cells, S means Number of Statements, P means Number of Parameters, KLCID means Kind of Line of Code Identifier Density, NBD means Nested Block Depth, OPRND means Number of Operands, OPRATOR means Number of Operators, UOPRND means Number of Unique Operands, UOPRATOR means Number of Unique Operators, ID means Number of Identifiers, ALID means Average Length of Identifiers, MLID means Max Length of Identifiers, CyC means Cyclomatic Complexity, EAP means External API Popularity, LOCom means Lines of Comments, CW means Number of Comment Words).\n\n"
            # + "\nLook at this python code, its code metrics, and its summary:\n"
            # + f"\n#Code:\n{sample_code}\n#Code Metrics:\n{sample_metrics}\n#Summary:\n{sample_summary}\n"
            # + "\nNow given this code and its code metrics, please give me the summary of the code\n"
            + f'\nCode 1: ```\n{sample_code}\n```\nCode Metrics 1: """\n{sample_metrics}\n"""\nSummary 1: """\n{sample_summary}\n"""\n'
            + f'\nCode 2: ```\n{task.question.code}\n```\nCode Metrics 2: """\n{metrics_string}\n"""\nSummary 2:'
        )




