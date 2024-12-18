import argparse
import transformers
from tqdm import tqdm
import matplotlib.pyplot as plt
from base import ExpBase, SRCS, MODELS
from dataclasses import dataclass
from gec_attribute import get_method, get_metric
import numpy as np
plt.rcParams["font.size"] = 15
transformers.logging.set_verbosity_error()

class ExpShapleySampling(ExpBase):
    output_dir = 'outputs/shapley_sampling'

    @dataclass
    class Config(ExpBase.Config):
        src: str = None
        hyps: list[str] = None
        metric: str = 'some'
        min_edit: int = 10
        max_edit: int = 15
    
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
        metric_cls = get_metric(config.metric)
        metric = metric_cls(metric_cls.Config())
        method_cls = get_method('shapley')
        self.shapley = method_cls(method_cls.Config(metric=metric))
        method_cls = get_method('shapleysampling')
        self.shapley_sampling = method_cls(method_cls.Config(
            metric=metric,
            num_samples=15  # T in the paper
        ))
        
    def run(self):
        shapley = {'score': [], 'time': []}
        shapley_sampling = {'score': [], 'time': []}
        for s, h in tqdm(zip(self.srcs, self.hyps)):
            edits = self.extract_edits(s, h)
            num_edits = len(edits)
            if not (self.config.min_edit <= num_edits <= self.config.max_edit):
                continue
            _ = self.time()
            out = self.shapley.attribute(s, inputs_edits=edits)
            t = self.time()
            shapley['score'] += out.attribution_scores
            shapley['time'].append(t)

            _ = self.time()
            out = self.shapley_sampling.attribute(s, inputs_edits=edits)
            t = self.time()
            shapley_sampling['score'] += out.attribution_scores
            shapley_sampling['time'].append(t)
        return {
            'shapley': shapley,
            'shapleysampling': shapley_sampling
        }

    def latexify(self, obj, **args):
        errors = [abs(s - ss) for s, ss in zip(obj['shapley']['score'], obj['shapleysampling']['score'])]
        avg_error = sum(errors) / len(errors)
        avg_time = sum(obj['shapleysampling']['time']) / len(obj['shapleysampling']['time'])
        # avg_time = sum(obj['shapley']['time']) / len(obj['shapley']['time'])
        orig = np.array([abs(s) for s in obj['shapley']['score']])
        avg = orig.mean()
        std = orig.std()
        print(avg_error, avg_time)
        return f'{self.config.metric.upper()} & {avg_error:.3f} & {avg_time:.2f} & {avg:.3f} ± {std:.3f}  \\\\\n'
    
def main(args):
    data = 'jfleg-dev'
    metrics = ['some', 'impara', 'ppl']
    src = SRCS[data]
    hyps = [f'experiments/corrected/{data}/{m}.txt' for m in MODELS]
    exp = None
    for metric in metrics:
        config = ExpShapleySampling.Config(
            src=src,
            hyps=hyps,
            metric=metric,
            min_edit=10,
            max_edit=15,
        )
        path = f'{metric}.json'
        if args.restore:
            exp = ExpShapleySampling(config)
            out = exp.load(path)
        else:
            if exp is None:
                exp = ExpShapleySampling(config)
            else:
                exp.reload(config)
            out = exp.run()
            exp.save(out, path)
        print(exp.latexify(out))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)