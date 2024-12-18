import argparse
import matplotlib.pyplot as plt
from collections import Counter
import itertools
from base import ExpBase, SRCS, solve_data_name, solve_model_name
from dataclasses import dataclass
import os


class ExpNumEditDistribution(ExpBase):
    output_dir = "outputs/num_edit_dist"

    @dataclass
    class Config(ExpBase.Config):
        src: str = None
        hyp: str = None

    def __init__(self, config):
        super().__init__(config)
        self.reload(config)

    def reload(self, config):
        self.config = config
        self.srcs = self.read_file(config.src)
        self.hyps = self.read_file(config.hyp)
        assert len(self.srcs) == len(self.hyps)

    def run(self):
        lengths = []
        for s, h in zip(self.srcs, self.hyps):
            edits = self.extract_edits(s, h)
            lengths.append(len(edits))
        length_dict = Counter(lengths)
        assert sum(length_dict.values()) == len(self.srcs)
        return length_dict

    def plot(self, length_dict, **args):
        total_sents = sum(length_dict.values())
        under6 = sum(length_dict.get(i, 0) for i in range(6))
        middle = [length_dict.get(i, 0) for i in range(6, 20)]
        over20 = sum(length_dict.get(i, 0) for i in range(20, max(length_dict.keys())))
        if over20 == []:
            over20 = [0]
        _y = [under6] + middle + [over20]
        accumu_y = list(itertools.accumulate(_y))
        y = [x / total_sents for x in accumu_y]
        x = list(range(len(y)))
        plt.plot(x, y, **args)


def main(args):
    datas = ["conll14", "jfleg-dev"][:]
    systems = ["bart", "gector-2024", "gector-roberta", "t5", "gpt-4o-mini"][:]
    exp = None
    for data, sys in itertools.product(datas, systems):
        src = SRCS[data]
        config = ExpNumEditDistribution.Config(
            src=src, hyp=f"experiments/corrected/{data}/{sys}.txt"
        )
        path = f"{data}-{sys}.json"
        if args.restore:
            exp = ExpNumEditDistribution(config)
            out = exp.load(path)
        else:
            if exp is None:
                exp = ExpNumEditDistribution(config)
            else:
                exp.reload(config)
            out = exp.run()
            exp.save(out, path)

        style = {
            "bart": (0, (1, 0)),
            "t5": (0, (1, 1)),
            "gector-roberta": (0, (5, 1)),
            "gector-2024": (0, (5, 1, 1, 1)),
            "gpt-4o-mini": (0, (5, 3, 1, 3, 1, 3)),
        }
        exp.plot(
            out,
            color="b" if data == "conll14" else "g",
            linestyle=style[sys],
            label=f"{solve_model_name(sys)} ({solve_data_name(data)})",
        )
    # red line
    plt.plot([5, 5], [0.9, 1.01], color="r", alpha=0.5)
    plt.legend(fontsize=12)
    plt.xlabel("Number of edits", fontsize=16)
    plt.ylabel("Cumulative ratio", fontsize=16)
    plt.xticks(
        list(range(16)),
        ["<6"] + list(map(str, [l + 5 for l in list(range(1, 16 - 1))] + ["19<"])),
    )
    plt.ylim((0.9, 1.01))
    plt.yticks(
        [l / 100 for l in list(range(90, 102))], [l / 100 for l in list(range(90, 102))]
    )
    plt.grid(axis="y", alpha=0.5)
    plt.tight_layout()
    outdir = ExpNumEditDistribution.output_dir
    plt.savefig(os.path.join(outdir, "out.png"))
    plt.savefig(os.path.join(outdir, "out.pdf"))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()
    main(args)
