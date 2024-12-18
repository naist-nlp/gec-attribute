import abc
import errant
from typing import Optional, Union
from dataclasses import dataclass

class AttributionBase(abc.ABC):
    @dataclass
    class Config:
        metric: str = None
        max_num_edits: int = float('inf')
        errant_language: str = 'en'
        quiet: bool = True
        num_samples: int = 64  # T for the Shapley sampling values.

    @dataclass
    class AttributionOutput:
        sent_score: float = None
        src_score: float = None
        attribution_scores: list[float] = None
        edits: list[errant.edit.Edit] = None
        src: str = None

    def __init__(self, config: Config):
        self.config = config
        self.scorer = config.metric
        self.errant_annotator = errant.load(config.errant_language)

    @abc.abstractmethod
    def generate(
        self,
        src: str,
        edits: list[errant.edit.Edit]
    ) -> list[dict]:
        '''Generate edited sentence.
        
        Args:
            src (str): source sentence.
            edits (list[errant.edit.Edit]): Edit to be applied to the source.

        Returns:
            list[Dict]: Each element has two keys:
                "sentence": An edited sentence.
                "indices": Indices of edits that affect editing according to the setting.
        '''
    
    @abc.abstractmethod
    def post_process(
        self,
        scores: list[float],
        sent_level_score: Optional[float] = None,
        indices: Optional[list[tuple]] = None
    ) -> list[float]:
        '''Post processing depending on the method.
        E.g. normalize for one-by-one method or sum up for Shapley theory.

        Args:
            scores (list[float]): \delta M() scores.
            sent_level_score (Optional[float]): Used when normalization.
            indices (Optional[list[Tuple]]): Which edits were applied to the source.
        
        Returns:
            list[float]: Post pocessed scores.
        '''

    def attribute(
        self,
        src: str,
        hyp: Optional[str] = None,
        inputs_edits: Optional[list[errant.edit.Edit]] = None
    ) -> AttributionOutput:
        '''Calculate attribution scores.
        
        Args:
            src (str): A source sentence.
            hyp (Optional[str]): An edited sentence.
            inputs_edits (Optional[list[errant.edit.Edit]]): 
                An alternative way to pass the edited sentence, as edit objects.

        Returns:
            AttributionOutput: Attribution scores and related information.
        '''

        if inputs_edits is not None:
            edits = inputs_edits
            if edits != [] and isinstance(edits[0], list):
                hyp = self.apply_edits(
                    src,
                    [ee for e in edits for ee in e]
                )
            else:
                hyp = self.apply_edits(
                    src,
                    edits
                )
        else:
            assert hyp is not None
            assert self.errant_annotator is not None
            edits = self.errant_annotator.annotate(
                self.errant_annotator.parse(src),
                self.errant_annotator.parse(hyp)
            )
        empty_result = self.AttributionOutput(
            sent_score=0,
            src_score=0,
            attribution_scores=[],
            edits=[],
            src=src
        )
        if len(edits) > self.config.max_num_edits:
            if not self.config.quiet:
                print('too many edits:', len(edits))
            return empty_result
        if edits == []:
            return empty_result
        
        edited_sentences = self.generate(src, edits)
        sentences = [e['sentence'] for e in edited_sentences] + [src, hyp]
        scores = self.scorer.score_sentence(
            [src] * len(sentences),
            sentences
        )
        # Source's score M(S, S)
        src_score = scores[-2]
        # Corrected sentence's score (Sentence-level score) M(S, H)
        hyp_score = scores[-1]
        # \delta M() 
        sent_level_score = hyp_score - src_score
        scores = [s - src_score for s in scores[:-2]]
        
        attribution_scores = self.post_process(
            scores,
            sent_level_score,
            indices=[e['indices'] for e in edited_sentences]
        )
        
        return self.AttributionOutput(
            sent_score=sent_level_score,
            src_score=src_score,
            attribution_scores=attribution_scores,
            edits=edits,
            src=src
        )

    def apply_edits(
        self,
        src: str,
        edits: list[errant.edit.Edit]
    ) -> str:
        '''Edit the source by the edits.
        
        Args:
            src (str): An input source sentence.
            edits (list[Edit]): An edit sequence.
        
        Returns:
            str: An edited sentence.
        '''
        # Firstly sort edits by start index.
        edits = sorted(edits, key=lambda x:x.o_start)
        offset = 0
        tokens = src.split(' ')
        for e in edits:
            if e.o_start == -1:
                continue
            s_idx = e.o_start + offset
            e_idx = e.o_end + offset
            # Is deletion edit
            if e.c_str == '':
                tokens[s_idx:e_idx] = ['$DELETE']
                offset -= (e.o_end - e.o_start) - 1
            # Is insertion edit
            elif e.o_start == e.o_end:
                tokens[s_idx:e_idx] = e.c_str.split(' ')
                offset += len(e.c_str.split())
            # Otherwise replacement edit
            else:
                tokens[s_idx:e_idx] = e.c_str.split(' ')
                offset += len(e.c_str.split(' ')) - (e.o_end - e.o_start)
        trg = ' '.join(tokens).replace(' $DELETE', '').replace('$DELETE ', '')
        return trg
