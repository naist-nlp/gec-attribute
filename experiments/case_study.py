import argparse
import matplotlib.pyplot as plt
import json
from dataclasses import dataclass
from gec_attribute import get_metric, get_method
from base import ExpBase, SRCS, MODELS
import json
import itertools
import os

class ExpCaseStudy(ExpBase):
    output_dir = 'outputs/case-study'

    @dataclass
    class Config(ExpBase.Config):
        src: str = None
        hyp: str = None
        metric: str = 'some'
        method: str = 'shapley'
        max_num_edits: int = 10

    def __init__(self, config):
        super().__init__(config)
        self.reload(config)
        
    def reload(self, config):
        self.config = config
        self.srcs = self.read_file(config.src)
        self.hyps = self.read_file(config.hyp)
        metric_cls = get_metric(config.metric)
        metric = metric_cls(metric_cls.Config())
        method_cls = get_method(config.method)
        self.attributor = method_cls(method_cls.Config(
            metric=metric,
            max_num_edits=config.max_num_edits
        ))

    def l1norm(self, scores: list[float]) -> list[float]:
        '''Normalize the attribution scores at sentence-level
        '''
        abs_sum = sum(abs(score) for score in scores)
        norm_scores = [score / abs_sum for score in scores]
        return norm_scores  # (num_sents, num_edits)
    
    def chunker(self, src, edits, attr_scores, norm_scores):
        tokens = src.split(' ')
        chunks = []
        previous_position = 0
        for edit_id, e in enumerate(edits):
            if e.o_start != previous_position:
                chunks.append({
                    'o_str': ' '.join(tokens[previous_position: e.o_start]),
                    'c_str': ' '.join(tokens[previous_position: e.o_start]),
                    'attr_score': None,
                    'norm_score': None
                })
            chunks.append({
                'o_str': ' '.join(tokens[e.o_start: e.o_end]),
                'c_str': e.c_str,
                'attr_score': attr_scores[edit_id],
                'norm_score': norm_scores[edit_id]
            })
            previous_position = e.o_end
        if previous_position < len(tokens):
            chunks.append({
                'o_str': ' '.join(tokens[previous_position:]),
                'c_str': ' '.join(tokens[previous_position:]),
                'attr_score': None,
                'norm_score': None
            })
        return chunks
        
    def run(self):
        results = []
        for sent_id, (s, h) in enumerate(zip(self.srcs, self.hyps)):
            edits = self.extract_edits(s, h)
            output = self.attributor.attribute(s, inputs_edits=edits)
            dummy_res = {
                'id': sent_id,
                'sent_score': '(Num edits exceeding)',
                'chunks': None
            }
            if output.attribution_scores == []:
                results.append(dummy_res)
                continue
            if sum(output.attribution_scores) == 0:
                results.append(dummy_res)
                continue
            norm_scores = self.l1norm(output.attribution_scores)
            assert len(edits) == len(output.attribution_scores)
            assert len(edits) == len(norm_scores)
            chunks = self.chunker(
                s, edits,
                output.attribution_scores,
                norm_scores
            )
            results.append({
                'id': sent_id,
                'sent_score': output.sent_score,
                'chunks': chunks
            })
        return results
    
    def visualize(self, results, tex=False):
        def add_space(s: str, max_len: int):
            offset = max_len - len(s)
            num_space = offset//2
            if offset % 2 == 0:
                return ' ' * num_space + s + ' ' * num_space
            else:
                return ' ' * num_space + s + ' ' * (num_space + 1)
            
        ret = ''
        for res in results:
            ret += f"=== Sentence {res['id']} (Delta M() = {res['sent_score']}) ===\n"
            if res['chunks'] == None:
                ret += 'None \n\n\n\n\n'
                continue
            strings = [f'{"Original":10} |', f'{"Corrected":10} |', f'{"Score":10}|', f'{"Norm. score":10}|']
            for c in res['chunks']:
                attr_score = f"{c['attr_score']:.3f}" if c['attr_score'] is not None else '-'
                norm_score = f"{c['norm_score']:.3f}" if c['attr_score'] is not None else '-'
                max_len = max(len(c['o_str']), len(c['c_str']), len(attr_score), len(norm_score))
                strings[0] += add_space(c['o_str'], max_len) + '|'
                strings[1] += add_space(c['c_str'], max_len) + '|'
                strings[2] += add_space(attr_score, max_len) + '|'
                strings[3] += add_space(norm_score, max_len) + '|'
            for s in strings:
                if tex:
                    s = s[:-1].replace('|', ' & ')
                    ret += s + '\\\\\n'
                else:
                    ret += s + '\n'
            ret += '\n'
        return ret

def main(args):
    exp = None
    model = 't5'
    methods = ['shapley']
    metrics = ['some', 'impara', 'ppl']
    data = 'jfleg-dev'
    for method, metric in itertools.product(methods, metrics):
        config = ExpCaseStudy.Config(
            src=SRCS[data],
            hyp=f'experiments/corrected/{data}/{model}.txt',
            metric=metric,
            method=method
        )
        path = f"{data}-{method}-{metric}-{model}.json"
        if args.restore:
            exp = ExpCaseStudy(config)
            out = exp.load(path)
        else:
            if exp is None:
                exp = ExpCaseStudy(config)
            else:
                exp.reload(config)
            out = exp.run()
            exp.save(out, path)
        ret = exp.visualize(
            out,
            tex=True
        )
        path = os.path.join(ExpCaseStudy.output_dir, f"{data}-{method}-{metric}-{model}.txt")
        with open(path, 'w') as f:
            f.write(ret + '\n')
    

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()
    main(args)
