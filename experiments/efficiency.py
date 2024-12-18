import argparse
import transformers
from tqdm import tqdm
import matplotlib.pyplot as plt
from base import ExpBase, SRCS, MODELS
from dataclasses import dataclass
from gec_attribute import get_method, get_metric
import itertools
import os
transformers.logging.set_verbosity_error()
plt.rcParams["font.size"] = 15

class ExpEfficiency(ExpBase):
    output_dir = 'outputs/efficiency/'
    
    @dataclass
    class Config(ExpBase.Config):
        src: str = None
        hyps: list[str] = None
        method: str = 'shapley'
        metric: str = 'some'
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
        method_cls = get_method(config.method)
        self.attributor = method_cls(method_cls.Config(metric=metric))
        
    def run(self):
        num_edits2time = dict()
        print(len(self.srcs), len(self.hyps))
        for s, h in zip(self.srcs, self.hyps):
            edits = self.extract_edits(s, h)
            num_edits = len(edits)
            if num_edits > self.config.max_edit:
                continue
            _ = self.time()
            _ = self.attributor.attribute(s, inputs_edits=edits)
            t = self.time()
            num_edits2time[num_edits] = num_edits2time.get(num_edits, list())
            num_edits2time[num_edits].append(t)
        return {k: sum(v) / len(v) for k, v in num_edits2time.items()}

    def plot(self, num_edits2time, **args):
        x = list(range(self.config.max_edit))
        y = [num_edits2time.get(i, 0) for i in range(self.config.max_edit)]
        plt.plot(x, y, **args)

    def load(self, path):
        out = super().load(path)
        return {int(k): v for k, v in out.items()}

def main(args):
    data = 'jfleg-dev'
    methods = ['shapley']
    metrics = ['some', 'impara', 'ppl']
    src = SRCS[data]
    hyps = [f'experiments/corrected/{data}/{m}.txt' for m in MODELS]
    exp = None
    for method, metric in itertools.product(methods, metrics):
        config = ExpEfficiency.Config(
            src=src,
            hyps=hyps,
            method=method,
            metric=metric,
            max_edit=15
        )
        path = f'{config.max_edit}-{metric}.json'
        if args.restore:
            exp = ExpEfficiency(config)
            out = exp.load(path)
        else:
            if exp is None:
                exp = ExpEfficiency(config)
            else:
                exp.reload(config)
            out = exp.run()
            exp.save(out, path)
        exp.plot(
            out,
            label=metric.upper(),
        )
    plt.xlabel('Number of edits')
    plt.ylabel('Computation time (sec.)')
    plt.xticks(
        list(range(1, exp.config.max_edit, 2)),
        list(range(1, exp.config.max_edit, 2))
    )
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(os.path.join(ExpEfficiency.output_dir, 'out.png'))
    plt.savefig(os.path.join(ExpEfficiency.output_dir, 'out.pdf'))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)
