from typing import List, Dict, Optional, TYPE_CHECKING
from ise_cdg_utility.metrics.enums import CodeMetric
from ise_cdg_utility.metrics.interface import NLPMetricInterface
from ise_cdg_data.summarize import get_sumy_summarizer

if TYPE_CHECKING:
    from ise_cdg_utility.metrics.interface.src import NLPMetricTorchmetrics


class LLMNLPMetricAdaptor(NLPMetricInterface):
    def __init__(
        self, p95_len: Optional[int], nlp_metric: "NLPMetricTorchmetrics"
    ) -> None:
        self.p95_len = p95_len
        self.nlp_metric = nlp_metric
        self.summarizer = get_sumy_summarizer(2)

    def set_references(self, references: List[str]) -> None:
        self.nlp_metric.set_references(references)

    def __setattr__(self, name, value):
        if name == "use_tqdm":
            setattr(self.nlp_metric, name, value)
            return
        return super().__setattr__(name, value)

    def __call__(self, candidates: List[str]):
        _summarized = 0
        new_candidates = []
        for i in range(len(candidates)):
            candid = candidates[i]
            if self.p95_len is not None and len(candid) > self.p95_len:
                candid = self.summarizer(candid)
                _summarized += 1
            new_candidates.append(candid)
        if _summarized > 0:
            print(
                f"Warning: summarized {_summarized} candidates for {self.nlp_metric.__class__.__name__}!"
            )
        return self.nlp_metric(new_candidates)


def get_metrics(p95_md_len: Optional[int]) -> Dict[CodeMetric, LLMNLPMetricAdaptor]:
    from ise_cdg_utility.metrics.src import (
        NLPMetricRangedBLEU,
        NLPMetricROUGE,
        NLPMetricBERT,
    )

    def get_metric(metric):
        return LLMNLPMetricAdaptor(p95_md_len, metric)

    return {
        CodeMetric.BLEU: get_metric(NLPMetricRangedBLEU()),
        CodeMetric.ROUGE: get_metric(NLPMetricROUGE()),
        CodeMetric.BERT: get_metric(NLPMetricBERT()),
    }
