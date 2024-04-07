from typing import List, Dict
from ise_cdg_utility.metrics.enums import CodeMetric
from ise_cdg_utility.metrics.interface import NLPMetricInterface, VectorizedNLPMetric
from ise_cdg_data.tokenize import get_source_and_markdown_tokenizers
from ise_cdg_data.summarize import get_sumy_summarizer


class LLMNLPMetricAdaptor(VectorizedNLPMetric):
    def __init__(self, p95_len: int, nlp_metric: NLPMetricInterface) -> None:
        self.p95_len = p95_len
        self.nlp_metric = nlp_metric
        _, self.md_tokenizer = get_source_and_markdown_tokenizers(cleanse_markdown=False)
        self.summarizer = get_sumy_summarizer(2)

    def set_references(self, references: List[str]) -> None:
        new_references = []
        for i in range(len(references)):
            ref = references[i]
            reference = self.md_tokenizer(ref)
            new_references.append([reference])
        self.nlp_metric.set_references(new_references)

    def __call__(self, candidates: List[str]):
        new_candidates = []
        for i in range(len(candidates)):
            candid = candidates[i]
            if len(candid) > self.p95_len:
                candid = self.summarizer(candid)
            candidate = self.md_tokenizer(candid)
            new_candidates.append(candidate)
        return self.nlp_metric(new_candidates)



def get_metrics() -> Dict[CodeMetric, LLMNLPMetricAdaptor]:
    from ise_cdg_utility.metrics.src import NLPMetricRangedBLEU, NLPMetricROUGE, NLPMetricBERT
    def get_metric(metric):
        return LLMNLPMetricAdaptor(1200, metric)

    return {
        CodeMetric.BLEU: get_metric(NLPMetricRangedBLEU()),
        CodeMetric.ROUGE: get_metric(NLPMetricROUGE()),
        CodeMetric.BERT: get_metric(NLPMetricBERT()),
    }