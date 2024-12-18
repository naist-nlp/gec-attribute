import abc
from dataclasses import dataclass

class MetricBase(abc.ABC):
    @dataclass
    class Config: ...

    def __init__(self, config: Config):
        self.config = config
    
class MetricBaseForReferenceFree(MetricBase, metaclass=abc.ABCMeta):
    
    @dataclass
    class Config(MetricBase.Config): ...

    def score_corpus(
        self,
        sources: list[str],
        hypotheses: list[str]
    ) -> float:
        '''Calculate a corpus-level score.
        By default, it is defined as an average of sentence-level scores.

        Args:
            sources (list[str]): Source sentence.
            hypothesis (list[str]): Corrected sentences.
        
        Returns:
            float: The corpus-level score.
        '''
        scores = self.score_sentence(
            sources=sources,
            hypotheses=hypotheses
        )
        return sum(scores) / len(scores)
        
        
    @abc.abstractmethod
    def score_sentence(
        self,
        sources: list[str],
        hypotheses: list[str]
    ) -> list[float]:
        '''Calculate a sentence-level score.

        Args:
            sources (list[str]): Source sentence.
            hypothesis (list[str]): Corrected sentences.
        
        Returns:
            list[float]: The sentence-level scores.
        '''
        raise NotImplementedError