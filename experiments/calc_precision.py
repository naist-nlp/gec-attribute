import argparse
import transformers
import matplotlib.pyplot as plt
from dataclasses import dataclass
from base import ExpBase, SRCS, MODELS, solve_model_name
from gec_attribute import get_method, get_metric
import itertools
import os
transformers.logging.set_verbosity_error()
plt.rcParams["font.size"] = 15

class ExpPrecision(ExpBase):
    output_dir = 'outputs/precision/'

    @dataclass
    class Config:
        cat: int = 2  # 1: R/M/U, 2: NOUN/VERB, 3: R:NOUN/M:VERB
        src: str = None
        hyps: list[str] = None
        metric: str = 'some'
        method: str = 'shapley'
        max_num_edits: int = 10
    
    def __init__(self, config):
        super().__init__(config)
        self.reload(config)

    def reload(self, config):
        self.config = config
        self.srcs = self.read_file(config.src) * len(config.hyps)
        self.hyps = []
        for path in config.hyps:
            sents = self.read_file(path)
            self.hyps += sents
        assert len(self.srcs) == len(self.hyps)
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
    
    def run(self):
        results = []
        for sent_id, (s, h) in enumerate(zip(self.srcs, self.hyps)):
            edits = self.extract_edits(s, h)
            output = self.attributor.attribute(s, inputs_edits=edits)
            if output.attribution_scores == []:
                # The case of exceeding the number of edits.
                continue
            if sum(output.attribution_scores) == 0:
                continue
            norm_scores = self.l1norm(output.attribution_scores)
            results.append({
                'attribution-scores': output.attribution_scores,
                'normalize-scores': norm_scores,
                'types': [e.type for e in edits]
            })
        return results
    
    def plot(self, data_list):
        # Classify the normalized scores by error types.
        aggregate_data = dict()
        for k in data_list.keys():
            etype2scores = dict()
            for d in data_list[k]:
                for norm_s, t in zip(d['normalize-scores'], d['types']):
                    if self.config.cat == 1:
                        t = t[0]
                    elif self.config.cat == 2:
                        t = t[2:]
                    etype2scores[t] = etype2scores.get(t, list())
                    etype2scores[t].append(norm_s)
            aggregate_data[k] = etype2scores
        
        corpus_score = {'gector-roberta': '0.802', 'gector-2024': '0.812', 'bart': '0.789', 't5': '0.803', 'gpt-4o-mini': '0.841'}
        metrics = ['bart', 'gector-roberta', 't5', 'gector-2024', 'gpt-4o-mini']
        num_metrics = len(data_list)
        etypes = sorted(list(aggregate_data[metrics[0]].keys()))
        num_etypes = len(etypes)
        
        # Create heatmap using error types with 30 or more frequency.
        heatmap = [list() for _ in range(num_metrics)]  # will be (num_metrics, num_error_typs)
        threshold_freq = 10
        visible_types = []
        for type_id, t in enumerate(etypes):
            min_freq = min(len(aggregate_data[m].get(t, [])) for m in metrics)
            if threshold_freq <= min_freq:
                visible_types.append(t)
                for metric_id, metric in enumerate(metrics):
                    score = aggregate_data[metric][t]
                    tp = sum(s for s in score if s > 0)
                    fp = sum(abs(s) for s in score if s < 0)
                    heatmap[metric_id].append(tp / (tp + fp))
        assert len(heatmap) == num_metrics

        # Let's plot
        heat = plt.pcolor(heatmap, cmap=plt.cm.Blues)
        plt.colorbar(heat)
        plt.yticks(
            [i + 0.5 for i in range(num_metrics)],
            [solve_model_name(m) + f' ({corpus_score[m]})' for m in metrics],
            rotation=45
        )
        plt.xticks(
            [i + 0.5 for i in range(len(visible_types))],
            visible_types,
            rotation=-75
        )
        plt.xlabel('Error types', fontsize=16)
        plt.ylabel('Models', fontsize=16)

def main(args):
    data_list = dict()
    exp = None
    data = 'jfleg-dev'
    src = SRCS[data]
    metrics = ['some']
    methods = ['shapley']
    for model in MODELS:
        for metric, method in itertools.product(metrics, methods):
            hyps = [f'experiments/corrected/{data}/{model}.txt']
            config = ExpPrecision.Config(
                cat=2,
                src=src,
                hyps=hyps,
                metric=metric,
                method=method
            )
            path = f'{data}-{method}-{metric}-{model}.json'
            if args.restore:
                exp = ExpPrecision(config)
                out = exp.load(path)
            else:
                if exp is None:
                    exp = ExpPrecision(config)
                else:
                    exp.reload(config)
                out = exp.run()
                exp.save(out, path)
            data_list[model] = out
    plt.figure(figsize=(10, 4))
    exp.plot(data_list)
    plt.tight_layout()
    plt.savefig(os.path.join(ExpPrecision.output_dir, "out.png"))
    plt.savefig(os.path.join(ExpPrecision.output_dir, "out.pdf"))
    plt.close()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)