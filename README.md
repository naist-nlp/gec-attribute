# Improving Explainability of Sentence-level Metrics via Edit-level Attribution for Grammatical Error Correction

This is the official repository for our [paper](https://arxiv.org/abs/2412.13110): 
```
@misc{goto2024improvingexplainabilitysentencelevelmetrics,
      title={Improving Explainability of Sentence-level Metrics via Edit-level Attribution for Grammatical Error Correction}, 
      author={Takumi Goto and Justin Vasselli and Taro Watanabe},
      year={2024},
      eprint={2412.13110},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.13110}, 
}
```

# Install
```sh
pip install git+https://github.com/naist-nlp/gec-attribute
python -m spacy download en_core_web_sm
```
Or
```sh
git clone https://github.com/naist-nlp/gec-attribute
cd gec-attribute
pip install -e ./
python -m spacy download en_core_web_sm
```

# Usage

- We have attribution classes and metric classes. First, create an instance of the metric, then create an instance of the attribution method using it as an argument.
    - Use `get_metric()` to get a metric class.
    - Use `get_method()` to get an attribution method class.
- You can use `.attribute()` to run an attribution and then get a `gec_attribution.method.AttributionBase.AttributionOutput` object.

```python
from gec_attribute import get_metric, get_method
from gec_attribute.methods import AttributionBase
import pprint

metric_cls = get_metric('impara')
metric = metric_cls(metric_cls.Config())
method_cls = get_method('shapley')
attributor = method_cls(method_cls.Config(
    metric=metric
))
# This is the example of Table 1 in our paper.
src = 'Further more by these evidence u will agree'
hyp = 'Further more , with this evidence , you will agree .'
output = attributor.attribute(src=src, hyp=hyp)
assert isinstance(output, AttributionBase.AttributionOutput)
pprint.pprint(output)
# Output:
# AttributionOutput(sent_score=-0.027204984799027443,
#                   src_score=0.027204984799027443,
#                   attribution_scores=[0.06834496899197498,
#                                       0.02928952643026908,
#                                       0.12393252272158858,
#                                       0.14501388886322578,
#                                       -0.36118950191885235,
#                                       -0.03259630793084701],
#                   edits=[<errant.edit.Edit object at 0x7fb3dc599890>,
#                          <errant.edit.Edit object at 0x7fb2f8339d90>,
#                          <errant.edit.Edit object at 0x7fb3dc5cf610>,
#                          <errant.edit.Edit object at 0x7fb2dc0ac9d0>,
#                          <errant.edit.Edit object at 0x7fb2dc0ac990>,
#                          <errant.edit.Edit object at 0x7fb2dc0ac890>],
#                   src='Further more by these evidence u will agree')
```

### Metrics
You can see the ids for `get_metric()` via `get_metric_ids()`.:
```python
from gec_attribute import get_metric_ids
print(get_metric_ids())
```

Currently these keys are available.
- `some`: SOME [[Yoshimura+ 20]](https://aclanthology.org/2020.coling-main.573/). Note that you need to download pre-trained models in advance from [here](https://github.com/kokeman/SOME#:~:text=Download%20trained%20model).
- `impara`: IMPARA [[Maeda+ 22]](https://aclanthology.org/2022.coling-1.316/).
- `ppl`: PPL.

### Attribution methods
You can see the ids for `get_method()` via `get_method_ids()`.:
```python
from gec_attribute import get_method_ids
print(get_method_ids())
```

Currently these keys are available.
- `add`: Add, one of the baselines.
- `sub`: Sub, one of the baselines.
- `shapley`: Shapley values, the proposed method.
- `shapleysampling`: Shapley sampling values.


# Reproduction of our experiments
The experimental scripts are available in [experiments/](experiments/).

Note that `pip install git+...` does not install these scripts. You need to do `git clone` instead.

### Data
The `experiments/corrected/` directory contains the corrected sentences used in our experiments.
```
experiments/corrected/
├── conll14
│   ├── bart.txt
│   ├── gector-2024.txt
│   ├── gector-roberta.txt
│   ├── gpt-4o-mini.txt
│   └── t5.txt
└── jfleg-dev
    ├── bart.txt
    ├── gector-2024.txt
    ├── gector-roberta.txt
    ├── gpt-4o-mini.txt
    └── t5.txt
```
