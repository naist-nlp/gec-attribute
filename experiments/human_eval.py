import argparse
from gec_attribute import get_method, get_metric
import errant
from typing import List
from errant.edit import Edit
from sklearn.metrics import classification_report, auc, matthews_corrcoef, accuracy_score, confusion_matrix
import transformers
from collections import Counter
import json
import matplotlib.pyplot as plt
import itertools
from dataclasses import dataclass
from base import ExpBase, SRCS, REFS, solve_model_name, METHODS, METRICS, MODELS
from logging import getLogger, StreamHandler, DEBUG
from collections import Counter
import os
transformers.logging.set_verbosity_error()
plt.rcParams["font.size"] = 15

class ExpHumanEvaluation(ExpBase):
    output_dir = 'outputs/human-eval'
    
    @dataclass
    class Config:
        src: str = None
        hyps: list[str] = None
        refs: list[str] = None
        metric: str = 'some'
        method: str = 'shapley'

    def __init__(self, config):
        super().__init__(config)
        self.logger = getLogger(__name__)
        handler = StreamHandler()
        handler.setLevel(DEBUG)
        self.logger.setLevel(DEBUG)
        self.logger.addHandler(handler)
        self.logger.propagate = False
        self.reload(config)

    def reload(self, config):
        self.srcs = self.read_file(config.src) * len(config.hyps)
        self.hyps = []
        for path in config.hyps:
            self.hyps += self.read_file(path)
        self.refs = [self.read_file(r) * len(config.hyps) for r in config.refs]
        assert len(self.srcs) == len(self.hyps)
        assert len(self.srcs) == len(self.refs[0])
        metric_cls = get_metric(config.metric)
        metric = metric_cls(metric_cls.Config())
        method_cls = get_method(config.method)
        self.attributor = method_cls(method_cls.Config(
            metric=metric,
            max_num_edits=10
        ))
        self.thresholds = self.get_thresholds(bins=10)

    def get_ref_base_labels(
        self,
        hyp_edits: list[Edit],  # (num_edits)
        ref_edits: list[list[Edit]],  # (num_refs, num_edits),
        attr_labels: list[int]  # (num_edits)
    ) -> list[list[int]]:  # (num_refs, num_edits)
        hyp = [(e.o_start, e.o_end, e.c_str) for e in hyp_edits]
        refs = [[(e.o_start, e.o_end, e.c_str) for e in r] for r in ref_edits]
        labels = None
        best_score = -999
        for ref in refs:
            # hyp--ref matching
            current_labels = [int(h in ref) for h in hyp]
            # attr--(hyp--ref) mathing
            acc = accuracy_score(current_labels, attr_labels)
            if best_score <= acc:
                best_score = acc
                labels = current_labels
        return labels
    
    def get_attr_labels(
        self,
        scores: list[float]  # (num_edits)
    ) -> list[int]:  # (num_edits)
        return [int(ss > 0) for ss in scores]
    
    def l1norm(self, scores: list[float]) -> list[float]:
        '''Normalize the attribution scores at sentence-level
        '''
        abs_sum = sum(abs(score) for score in scores)
        norm_scores = [score / abs_sum for score in scores]
        return norm_scores  # (num_sents, num_edits)
    
    def get_thresholds(self, bins=10):
        # [0.1, 0.2 ..., 1.0]
        return [i / bins for i in range(1, bins+1)]
    
    def run(self):
        num_sents = len(self.srcs)
        ref_labels = []
        attr_labels = []
        norm_scores = []
        for sent_id in range(num_sents):
            src = self.srcs[sent_id]
            hyp = self.hyps[sent_id]
            refs = [r[sent_id] for r in self.refs]
            hyp_edits = self.extract_edits(src, hyp)
            if len(hyp_edits) <= 1:
                continue
            ref_edits = [self.extract_edits(src, ref) for ref in refs]
            output = self.attributor.attribute(
                src, inputs_edits=hyp_edits
            )
            if output.attribution_scores == []:
                continue
            if sum(output.attribution_scores) == 0:
                continue
            current_attr_labels = self.get_attr_labels(
                output.attribution_scores
            )
            current_ref_labels = self.get_ref_base_labels(
                hyp_edits=hyp_edits,
                ref_edits=ref_edits,
                attr_labels=current_attr_labels
            )
            attr_labels += current_attr_labels
            ref_labels += current_ref_labels
            norm_scores += self.l1norm(output.attribution_scores)
            
        assert len(attr_labels) == len(ref_labels)
        assert len(attr_labels) == len(norm_scores)
        total_edits = len(norm_scores)
        self.logger.debug(f'{total_edits=}')

        cls_reports = []
        # Calculate accuacy by changing the threshold.
        for thre_id, thre in enumerate(self.thresholds):
            range_s = 0
            range_e = thre
            filtered_ref = [l for i, l in enumerate(ref_labels) \
                            if range_s <= abs(norm_scores[i]) <= range_e]
            filtered_attr = [l for i, l in enumerate(attr_labels) \
                             if range_s <= abs(norm_scores[i]) <= range_e]
            cls_report = classification_report(
                y_true=filtered_ref,
                y_pred=filtered_attr,
                zero_division=1.0,
                output_dict=True
            )
            cls_report['ref-labels'] = filtered_ref
            cls_report['attr-labels'] = filtered_attr
            cls_report['threshold'] = thre
            self.logger.debug(f'{thre=}, {thre_id=}, {range_s=} {range_e=}, {Counter(filtered_ref)=}, {Counter(filtered_attr)=} {cls_report["accuracy"]=}')
            cls_reports.append(cls_report)
        return cls_reports
    
    def plot(self, cls_reports, **args):
        accs = [rep['accuracy'] for rep in cls_reports]
        x = list(range(len(accs)))
        plt.plot(x, accs, **args)

def run(args):
    models = MODELS
    metrics = METRICS
    methods = ['add', 'sub', 'shapley']
    datas = ['conll14', 'jfleg-dev']
    exp = None
    for data in datas:
        src = SRCS[data]
        refs = REFS[data]
        hyps = [f'experiments/corrected/{data}/{m}.txt' for m in models]
        for metric, method in itertools.product(metrics, methods):
            config = ExpHumanEvaluation.Config(
                src=src, hyps=hyps, refs=refs,
                metric=metric,
                method=method
            )
            path = f"{data}-{method}-{metric}-5sys.json"
            if args.restore:
                exp = ExpHumanEvaluation(config)
                out = exp.load(path)
            else:
                if exp is None:
                    exp = ExpHumanEvaluation(config)
                else:
                    exp.reload(config)
                out = exp.run()
                exp.save(out, path)
            linestyles = [':', '--', '-', '-.']
            exp.plot(
                out,
                label=f'{metric.upper()}-{method.capitalize()}',
                linestyle=linestyles[methods.index(method)],
                color=plt.rcParams['axes.prop_cycle'].by_key()['color'][metrics.index(metric)],
            )

        plt.xticks(list(range(len(exp.thresholds))), exp.thresholds, rotation=45)
        plt.yticks(rotation=45)
        plt.xlabel('Thresholds')
        plt.ylabel('Accuracy')
        plt.grid(alpha=0.5, axis='y')
        # plt.title(solve_data_name(data))
        if data == 'jfleg-dev':
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)
        plt.savefig(os.path.join(ExpHumanEvaluation.output_dir, f"{data}-5sys.png"), bbox_inches='tight')
        plt.savefig(os.path.join(ExpHumanEvaluation.output_dir, f"{data}-5sys.pdf"), bbox_inches='tight')
        plt.close()

            

def main(args):
    run(args)
    return

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='/cl/nldata/GEC/conll14/unofficial/conll14.src')
    parser.add_argument('--hyp')
    parser.add_argument('--attr_score')
    parser.add_argument('--ref', default='ref_m')
    parser.add_argument('--ref_chose', default='attr-best')
    parser.add_argument('--threshold', default='accumu')
    parser.add_argument('--pos_neg', default='pos')
    parser.add_argument('--out', default='outputs/ref_base/out.png')
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)
    # # gen_data(args)

    # data = {
    #     'conll14': json.load(open('outputs/ref_base/conll14-base.json')),
    #     'jfleg-dev': json.load(open('outputs/ref_base/jfleg-dev-base.json'))
    # }
    # # latexfy(conll14, jfleg)
    # for data_name in ['conll14', 'jfleg-dev'][1:]:
    #     for score_name in ['all', 'pos', 'neg'][1:]:
    #         # ref_base_threshold(conll14)
    #         # ref_base_threshold(
    #         #     data[data_name],
    #         #     data_name,
    #         #     score_name
    #         # )
    #         ref_base_threshold_etype(
    #             data[data_name],
    #             data_name,
    #             score_name
    #         )