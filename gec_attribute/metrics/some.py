from .base import MetricBase, MetricBaseForReferenceFree
from dataclasses import dataclass
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import math

class SOME(MetricBaseForReferenceFree):
    @dataclass
    class Config(MetricBaseForReferenceFree.Config):
        model_f: str = 'gfm-models/fluency/'
        model_g: str = 'gfm-models/grammer/'
        model_m: str = 'gfm-models/meaning/'
        weight_g: float = 0.55
        weight_f: float = 0.43
        weight_m: float = 0.02
        no_cuda: bool = False
        batch_size: int = 32
        max_length: int = 128

    def __init__(self, config: Config):
        super().__init__(config)
        self.model_f = AutoModelForSequenceClassification.from_pretrained(config.model_f).cuda()
        self.model_g = AutoModelForSequenceClassification.from_pretrained(config.model_g).cuda()
        self.model_m = AutoModelForSequenceClassification.from_pretrained(config.model_m).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_f)
        if not config.no_cuda and torch.cuda.is_available():
            self.model_f.cuda()
            self.model_g.cuda()
            self.model_m.cuda()
    
    def min_max_normalize(self, x, x_min=1, x_max=4):
        return (x - x_min) / (x_max - x_min)
    
    def score_sentence(
        self,
        sources: list[str],
        hypotheses: list[str]
    ) -> list[float]:
        batch_size = self.config.batch_size
        scores = []
        for i in range(math.ceil(len(sources) / batch_size)):
            tokenizer_args = {
                'max_length': self.config.max_length,
                'padding': "max_length",
                'truncation': True,
                'return_tensors': 'pt'
            }
            encode_f = self.tokenizer(
                hypotheses[i*batch_size:(i+1)*batch_size],
                **tokenizer_args
            )
            encode_g = self.tokenizer(
                hypotheses[i*batch_size:(i+1)*batch_size],
                **tokenizer_args
            )
            encode_m = self.tokenizer(
                sources[i*batch_size:(i+1)*batch_size],
                hypotheses[i*batch_size:(i+1)*batch_size],
                **tokenizer_args
            )
            encode_f = {k: v.to(self.model_f.device) for k, v in encode_f.items()}
            encode_g = {k: v.to(self.model_g.device) for k, v in encode_g.items()}
            encode_m = {k: v.to(self.model_m.device) for k, v in encode_m.items()}
            with torch.no_grad():
                scores_g = self.model_g(**encode_g).logits.view(-1).tolist()
                scores_g = [self.min_max_normalize(s) for s in scores_g]
                scores_f = self.model_f(**encode_f).logits.view(-1).tolist()
                scores_f = [self.min_max_normalize(s) for s in scores_f]
                scores_m = self.model_m(**encode_m).logits.view(-1).tolist()
                scores_m = [self.min_max_normalize(s) for s in scores_m]
            batch_scores = [
                self.config.weight_f * s_f \
                + self.config.weight_g * s_g \
                + self.config.weight_m * s_m \
                for s_f, s_g, s_m in zip(scores_f, scores_g, scores_m)
            ]
            scores += batch_scores
        return scores