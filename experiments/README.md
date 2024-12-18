You first need to set the paths for sources and references to `base.py`.
```python
SRCS = {
    "conll14": "your path",
    "jfleg-dev": "your path",
}
REFS = {
    "conll14": [
        "your path",
        ...,
    ],
    "jfleg-dev": [
        "your path",
        ...,
    ],
}
```

# CLI

Once the script is run, the results are saved in json, and can be loaded a second time with the --restore option. This is useful if you only want to redraw the plot.

### Figure 2 (Cumulative sentence ratio according to the number of edits)

```sh
python experiments/num_edits_dist.py
```

### Figure 3 (Consistency evaluation)

```sh
python experiments/consistency.py
```

### Figure 4 (Human evaluation)
```sh
python experiments/human_eval.py
```

### Table 1 (Case study)

Table 1 is 672-th example of the T5 output for JFLEG-dev. 
```sh
python experiments/case_study.py
```

### Correlation between reference-based and -free (reported in Section 4.4)
```sh
python experiments/corr_base_free.py
```

### Figure 5 (Efficiency)
```sh
python experiments/efficiency.py
```

### Table 2 (Error and time of Shapley sampling values)
```sh
python experiments/shapley_sampling.py
```

### Figure 6 (Biases)
```sh
python experiments/bias.py
```

### Figure 7 (Precisions)
```sh
python experiments/calc_precision.py
```

# The use of API
All scripts contain a class stating with `Exp`. I believe that these classes also work on your custom data.

Below is the common usage.
```python
from experiments import ExpNumEditDistribution
# The arguments of Config are different depending on the class
exp = ExpNumEditDistribution(ExpNumEditDistribution.Config(
    src="your path", hyp="your path"
))
# All classes have .run()
out = exp.run()
# Save the results via .save()
exp.save(out, 'save path')

# for figures
exp.plot(out)
# for tables
exp.latexify(out)
# for others
exp.show(out)
```

Note that `.plot()` has only minimum function, e.g. plot a line.  
You can control detail settings, like subplot or labels, by adding such methods before or after `.plot()`.
```python
# Just example
plt.subplot(131)
exp.plot(out)
plt.title('sample title')
plt.xlabel('sample xlabel')
plt.savefig('sample.png')
```

`.latexify()` also shows only a row of the table, so you need to prepare header, e.g. `\begin{table}...` in your Tex file in advance, then insert the results of `latexify()`.

But different Exp classes have slightly different behavors, please refer to each script to check how to use them.