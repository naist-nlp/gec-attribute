from gec_metrics.metrics import ERRANT
import gec_attribute
import itertools
from base import ExpBase, SRCS, REFS
from dataclasses import dataclass
import argparse
        
class ExpCorrBase2Free(ExpBase):
    output_dir = 'outputs/corr-base-free'
    
    @dataclass
    class Config:
        metric: str = 'some'
        data: str = 'conll14'

    def __init__(self, config):
        super().__init__(config)
        self.metric_errant = ERRANT(ERRANT.Config())
        self.reload(config)

    def reload(self, config):
        self.config = config
        metric_cls = gec_attribute.get_metric(self.config.metric)
        self.metric = metric_cls(metric_cls.Config())
        self.data = self.load_data(config.data)

    def load_data(self, name):
        data = dict()
        data['sources'] = self.read_file(SRCS[name])
        data['references'] = [self.read_file(r) for r in REFS[name]]
        data['models'] = ['bart', 'gector-2024', 'gector-roberta', 'gpt-4o-mini', 't5']
        hyp_paths = [f'experiments/corrected/{name}/{m}.txt' \
                     for m in data['models']]
        data['hypotheses'] = [self.read_file(h) for h in hyp_paths]
        return data

    def calc_score_free(self):
        data = self.data
        scores = []
        for hyps in data['hypotheses']:
            scores.append(self.metric.score_sentence(
                sources=data['sources'],
                hypotheses=hyps
            ))
        return scores # (num_sys, num_sents)
    
    def calc_score_base(self):
        data = self.data
        scores = []
        for hyp in data['hypotheses']:
            scores.append(self.metric_errant.score_sentence(
                sources=data['sources'],
                hypotheses=hyp,
                references=data['references']
            ))
        return scores
    
    def acc(self, score1, score2):
        num_sys = len(score1)
        num_sents = len(score1[0])
        indices = list(range(num_sys))
        a, b = 0, 0
        for sent_id in range(num_sents):
            current_score1 = [score1[sys_id][sent_id] for sys_id in range(num_sys)]
            current_score2 = [score2[sys_id][sent_id] for sys_id in range(num_sys)]
            for i1, i2 in itertools.combinations(indices, 2):
                if current_score1[i1] == current_score1[i2]:
                    continue
                comp1 = current_score1[i1] > current_score1[i2]
                comp2 = current_score2[i1] > current_score2[i2]
                b += 1
                if comp1 == comp2:
                    a += 1
        acc = a / b
        kendall = (a - (b-a)) / b
        return acc, kendall
        
    def run(self):
        score_free = self.calc_score_free()
        score_base = self.calc_score_base()
        assert len(score_base) == len(score_free)
        assert len(score_base[0]) == len(score_free[0])
        return {
            'free': score_free,
            'base': score_base
        }
    
    def show(self, out):
        acc, kendall = self.acc(out['base'], out['free'])
        print(f'{acc=}, {kendall=}')
        

def main(args):
    exp = ExpCorrBase2Free(ExpCorrBase2Free.Config(
        metric='some',
        data='conll14'
    ))
    exp = None
    for data in ['conll14', 'jfleg-dev']:
        for metric in ['some', 'impara', 'ppl'][:]:
            config = ExpCorrBase2Free.Config(
                metric=metric,
                data=data
            )
            path = f'{data}-{metric}.json'
            if args.restore:
                exp = ExpCorrBase2Free(config)
                out = exp.load(path)
            else:
                if exp is None:
                    exp = ExpCorrBase2Free(config)
                else:
                    exp.reload(config)
                out = exp.run()
                exp.save(out, path)
            print(data, metric)
            exp.show(out)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)