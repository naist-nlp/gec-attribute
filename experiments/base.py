import argparse
import errant
import transformers
import matplotlib.pyplot as plt
from dataclasses import dataclass
import hashlib
import spacy
import json
import abc
import time
import os

transformers.logging.set_verbosity_error()

MODELS = ["gector-roberta", "gector-2024", "bart", "t5", "gpt-4o-mini"]
METRICS = ["some", "impara", "ppl"]
METHODS = ["add", "sub", "shapley"]

SRCS = {
    "conll14": "",
    "jfleg-dev": "",
}
REFS = {
    "conll14": [],
    "jfleg-dev": [],
}

class ExpBase(abc.ABC):
    output_dir = "outputs/base"

    @dataclass
    class Config: ...

    def __init__(self, config):
        self.config = config
        self.cache_parse = dict()
        self.cache_annotate = dict()
        self.errant = errant.load("en")
        self.current_time = 0

    def reload(self, config):
        """This is another __init__() while keeping errant cache."""
        pass

    def run(self):
        return {"message": "sample output"}

    def read_file(self, path: str) -> list[str]:
        sents = open(path).read().rstrip().split("\n")
        return sents

    def cached_parse(self, sent: str) -> spacy.tokens.doc.Doc:
        """Efficient parse() by caching.

        Args:
            sent (str): The sentence to be parsed.
        Return:
            spacy.tokens.doc.Doc: The parse results.
        """
        key = hashlib.sha256(sent.encode()).hexdigest()
        if self.cache_parse.get(key) is None:
            self.cache_parse[key] = self.errant.parse(sent)
        return self.cache_parse[key]

    def extract_edits(self, src: str, trg: str) -> list[errant.edit.Edit]:
        """Extract edits given a source and a corrected.

        Args:
            src (str): The source sentence.
            trg (str): The corrected sentence.

        Returns:
            list[errant.edit.Edit]: Extracted edits.
        """
        key = hashlib.sha256((src + "|||" + trg).encode()).hexdigest()
        if self.cache_annotate.get(key) is None:
            self.cache_annotate[key] = self.errant.annotate(
                self.cached_parse(src), self.cached_parse(trg)
            )
        return self.cache_annotate[key]

    def time(self):
        now = time.time()
        t = now - self.current_time
        self.current_time = now
        return t

    def save(self, object, path):
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, path)
        with open(path, "w") as fp:
            json.dump(object, fp, indent=2)
        return

    def load(self, path):
        path = os.path.join(self.output_dir, path)
        data = json.load(open(path))
        return data


def solve_model_name(name):
    return {
        "gector-roberta": "GECToR-RoBERTa",
        "gector-2024": "GECToR-2024",
        "t5": "T5",
        "bart": "BART",
        "gpt-4o-mini": "GPT-4o mini",
    }[name]


def solve_data_name(name):
    return {"conll14": "CoNLL-2014", "jfleg-dev": "JFLEG"}[name]


# Template
import argparse


def main(args):
    exp = None
    config = ExpBase.Config()
    path = "sample.json"
    if args.restore:
        exp = ExpBase(config)
        out = exp.load(path)
    else:
        if exp is None:
            exp = ExpBase(config)
        else:
            exp.reload(config)
        out = exp.run()
        exp.save(out, path)

    plt.savefig(os.path.join(ExpBase.output_dir, "out.png"))
    plt.savefig(os.path.join(ExpBase.output_dir, "out.pdf"))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()
    main(args)
