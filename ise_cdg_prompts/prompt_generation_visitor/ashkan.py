from typing import TYPE_CHECKING, List
from ise_cdg_prompts.prompt_generation_visitor.main import PromptGenerationVisitor
from ise_cdg_data.dataset.features_extractor import get_source_features_extractor
import pandas as pd

if TYPE_CHECKING:
    from ise_cdg_prompts.dataset import CodeMarkdown
    from ise_cdg_prompts.task import Task


class AshkanPromptGenerator(PromptGenerationVisitor):
    def visit_task(self, task: "Task") -> str:
        sentences = [
            "Hey falcon.",
            "Your going to write documention for a python code snippet.",
        ]

        shots_count = len(task.templates)
        if shots_count:
            if shots_count > 1:
                sentences += [
                    f"Initially, I've prepared for you {shots_count} examples each consisting of a code snippet and its corresponding documentaion.",
                    "Read them and learn how to write good documentation.\n",
                    "After these examples the query code is given and its summary is empty which you are going to fill.\n",
                ]
            else:
                sentences += [
                    f"Initially, I've prepared for you one example consisting of a code snippet and its corresponding documentaion.",
                    "Read it and learn how to write good documentation.\n",
                    "After this example the query code is given and its summary is empty which you are going to fill.\n",
                ]

        return (
            "".join(sentences)
            + self.visit_templates(task.templates)
            + f"Query:\nCode:\n{task.question.code}\nSummary:\n"
        )

    def visit_templates(self, templates: List["CodeMarkdown"]) -> str:
        return "".join(
            [
                self.visit_template(template, index)
                for index, template in enumerate(templates)
            ]
        )

    def visit_template(self, template: "CodeMarkdown", index: int) -> str:
        return f"Example {index + 1}:\nCode:\n{template.code}\nSummary:\n{template.markdown}\n"


class AshkanPromptGeneratorWithCodeMetric(PromptGenerationVisitor):
    def _get_metric_string(self, source):
        df = pd.DataFrame({"source": [source]})
        get_source_features_extractor().extract_feature_columns(code_df=df)
        df.drop(columns=["API", "source"], inplace=True)
        metrics_map = df.to_dict(orient="records")[0]
        metrics_string = ", ".join(
            [f"{name}: {value}" for name, value in metrics_map.items()]
        )
        return metrics_string

    def visit_task(self, task: "Task") -> str:
        metrics_string = self._get_metric_string(task.question.code)
        sentences = [
            "Hey falcon.",
            "Your going to write documention for a python code snippet.",
        ]

        shots_count = len(task.templates)
        if shots_count:
            if shots_count > 1:
                sentences += [
                    f"Initially, I've prepared for you {shots_count} examples each consisting of a code snippet, its underlying code metrics, and the corresponding documentaion.",
                    "Here is the meaning of each code metric used for each code snippet: LOC means Lines of Code, BLC means Number of Blank Lines of Code, UDF means Number of User-Defined Functions, I means Number of Imports, EH means Number of error handlings, ALLC means Average Line Length of Code, NDD means Number of Visualization Data Types, NEC means Number of Executed Cells, S means Number of Statements, P means Number of Parameters, KLCID means Kind of Line of Code Identifier Density, NBD means Nested Block Depth, OPRND means Number of Operands, OPRATOR means Number of Operators, UOPRND means Number of Unique Operands, UOPRATOR means Number of Unique Operators, ID means Number of Identifiers, ALID means Average Length of Identifiers, MLID means Max Length of Identifiers, CyC means Cyclomatic Complexity, EAP means External API Popularity, LOCom means Lines of Comments, CW means Number of Comment Words.",
                    "Read them and learn how to write good documentation.\n",
                    "After these examples the query code is given alongside its code metrics and its summary is empty which you are going to fill.\n",
                ]
            else:
                sentences += [
                    f"Initially, I've prepared for you one example consisting of a code snippet, its corresponding documentaion, and the underlying code metrics.",
                    "Here is the meaning of each code metric used for each code snippet: LOC means Lines of Code, BLC means Number of Blank Lines of Code, UDF means Number of User-Defined Functions, I means Number of Imports, EH means Number of error handlings, ALLC means Average Line Length of Code, NDD means Number of Visualization Data Types, NEC means Number of Executed Cells, S means Number of Statements, P means Number of Parameters, KLCID means Kind of Line of Code Identifier Density, NBD means Nested Block Depth, OPRND means Number of Operands, OPRATOR means Number of Operators, UOPRND means Number of Unique Operands, UOPRATOR means Number of Unique Operators, ID means Number of Identifiers, ALID means Average Length of Identifiers, MLID means Max Length of Identifiers, CyC means Cyclomatic Complexity, EAP means External API Popularity, LOCom means Lines of Comments, CW means Number of Comment Words.",
                    "Read it and learn how to write good documentation.\n",
                    "After these examples the query code is given alongside its code metrics and its summary is empty which you are going to fill.\n",
                ]

        return (
            "".join(sentences)
            + self.visit_templates(task.templates)
            + f"Query:\nCode:\n{task.question.code}\nCode Metrics:\n{metrics_string}\nSummary:\n"
        )

    def visit_templates(self, templates: List["CodeMarkdown"]) -> str:
        return "".join(
            [
                self.visit_template(template, index)
                for index, template in enumerate(templates)
            ]
        )

    def visit_template(self, template: "CodeMarkdown", index: int) -> str:
        metrics_string = self._get_metric_string(template.code)
        return f"Example {index + 1}:\nCode:\n{template.code}\nCode Metrics:\n{metrics_string}\nSummary:\n{template.markdown}\n"
