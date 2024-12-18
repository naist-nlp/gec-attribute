from .base import MetricBaseForReferenceFree
import lmppl
import numpy as np
from dataclasses import dataclass

class PPL(MetricBaseForReferenceFree):
    @dataclass
    class Config(MetricBaseForReferenceFree.Config):
        model: str = 'gpt2'
        batch_size: int = 32

    def __init__(self, config: Config):
        super().__init__(config)
        self.scorer = lmppl.LM(config.model)
    
    def score_sentence(
        self,
        sources: list[str],
        hypotheses: list[str]
    ) -> list[float]:
        scores = -np.array(self.scorer.get_perplexity(
            hypotheses,
            batch=self.config.batch_size
        ))
        scores = [s if str(s) != 'nan' else 0 for s in scores]
        return scores
