import argparse
from gec_attribute import get_method, get_metric
import errant
import transformers
from tqdm import tqdm
import pprint
from scipy.stats import spearmanr, pearsonr
import json
import matplotlib.pyplot as plt
import numpy as np
from base import ExpBase, SRCS, MODELS, METHODS, solve_model_name
from dataclasses import dataclass
import itertools
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
transformers.logging.set_verbosity_error()
plt.rcParams.update({'font.size': 20})

class ExpConsistency(ExpBase):
    output_dir = 'outputs/consistency'

    @dataclass
    class Config(ExpBase.Config):
        max_edit: int = 10
        method: str = 'shapley'
        metric: str = 'some'
        src: str = None
        hyps: list[str] = None

    def __init__(self, config):
        super().__init__(config)
        self.reload(config)
    
    def reload(self, config):
        self.srcs = self.read_file(config.src) * len(config.hyps)
        self.hyps = []
        for path in config.hyps:
            self.hyps += self.read_file(path)
        assert len(self.srcs) == len(self.hyps)
        metric_cls = get_metric(config.metric)
        metric = metric_cls(metric_cls.Config())
        method_cls = get_method(config.method)
        self.attributor = method_cls(method_cls.Config(metric=metric))
    
    def run(self):
        orig_scores = []
        grouped_scores = []
        for s, h in zip(self.srcs, self.hyps):
            edits = self.extract_edits(s, h)
            if len(edits) <= 2:
                continue
            if len(edits) > self.config.max_edit:
                continue
            attr_out = self.attributor.attribute(s, inputs_edits=edits)
            pos_count = sum(s > 0 for s in attr_out.attribution_scores)
            if pos_count != len(attr_out.attribution_scores):
                pos_edits, neg_edits = [], []
                orig_pos_score, orig_neg_score = 0, 0
                for edit_id, e in enumerate(edits):
                    attr_score = attr_out.attribution_scores[edit_id]
                    if attr_score > 0:
                        pos_edits.append(e)
                        orig_pos_score += attr_score
                    elif attr_score < 0:
                        neg_edits.append(e)
                        orig_neg_score += attr_score
                attr_out_grouped = self.attributor.attribute(
                    s, inputs_edits=[pos_edits, neg_edits]
                )
                g_scores = attr_out_grouped.attribution_scores
                assert len(g_scores) == 2
                if len(pos_edits) >= 2:
                    orig_scores.append(orig_pos_score)
                    grouped_scores.append(g_scores[0])
                if len(neg_edits) >= 2:
                    orig_scores.append(orig_neg_score)
                    grouped_scores.append(g_scores[1])
        assert len(orig_scores) == len(grouped_scores)
        return {
            'orig': orig_scores,
            'group': grouped_scores
        }
    
    def plot_sign(self, obj, **args):
        print(f"{len(obj['orig'])=}")
        is_match = [(o > 0) == (g > 0) for o, g in zip(obj['orig'], obj['group'])]
        ratio = sum(is_match) / len(is_match)
        assert 'x' in args
        plt.bar(height=ratio, **args)

    def plot_pearson(self, obj, **args):
        height = pearsonr(obj['orig'], obj['group'])[0]
        assert 'x' in args
        plt.bar(height=height, **args)

    def plot_spearman(self, obj, **args):
        height = spearmanr(obj['orig'], obj['group'])[0]
        assert 'x' in args
        plt.bar(height=height, **args)

def main_5sys(args):
    methods = METHODS
    models = MODELS
    exp = None
    offset = 0.5  # Add, Sub, Shapley spacing
    width = 0.1  # Bar's width
    datas = ['conll14', 'jfleg-dev'][:]
    metrics = ['some', 'impara', 'ppl'][:]
    for data in datas:
        plt.figure(figsize=(22, 3))
        for metric in metrics:
            plt.subplot(131 + metrics.index(metric))
            for method in methods[:]:
                src = SRCS[data]
                hyps = [f'experiments/corrected/{data}/{m}.txt' for m in models]
                config = ExpConsistency.Config(
                    max_edit=10,
                    metric=metric,
                    method=method,
                    src=src, hyps=hyps
                )
                path = f'{data}-{method}-{metric}-5sys.json'
                if args.restore:
                    exp = ExpConsistency(config)
                    out = exp.load(path)
                else:
                    if exp is None:
                        exp = ExpConsistency(config)
                    else:
                        exp.reload(config)
                    out = exp.run()
                    exp.save(out, path)
                x = methods.index(method) * width
                
                plot_config = {
                    'x': x,
                    'width': width,
                    'color': plt.rcParams['axes.prop_cycle'].by_key()['color'][7+methods.index(method)]
                }
                exp.plot_sign(out, **plot_config)
                plot_config['x'] += offset
                exp.plot_spearman(out, **plot_config)
                plot_config['x'] += offset
                if metrics.index(metric) == 1:
                    plot_config['label'] = method.capitalize()
                exp.plot_pearson(out, **plot_config)
            plt.xticks(
                [i * offset + width * 1 for i in range(3)],
                ['Sign (match ratio)', 'Mag. (Peason)', 'Mag. (Spearman)'],
                rotation=15
            )
            plt.grid(alpha=0.5, axis='y')
            plt.title(metric.upper())
        plt.subplot(132)  # center figure
        if data == 'jfleg-dev':
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.43), ncol=3)

        plt.savefig(os.path.join(ExpConsistency.output_dir, f"{data}-5sys.png"), bbox_inches='tight')
        plt.savefig(os.path.join(ExpConsistency.output_dir, f"{data}-5sys.pdf"), bbox_inches='tight')
        plt.close()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    # main(args)
    main_5sys(args)