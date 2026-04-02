from argparse import ArgumentParser
import json
from pathlib import Path

from pprint import pprint
import seqeval.metrics

from train_ner import TAGSETS, read_conll_file



parser = ArgumentParser()
parser.add_argument("--input_dir", "-i", type=Path, default=Path("./review"))
args = parser.parse_args()

with (args.input_dir / 'annotator_statistics.json').open() as f:
    data = json.load(f)

for language, stats in data.items():
    if len(stats) == 1:
        continue
    # pick the top 2 annotators
    stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:2]
    annotators = [stat[0] for stat in stats]

    # discover files annotated by both annotators
    files = {
        "annotator1": sorted((args.input_dir / "labelstudio_conll").glob(f"{language}*.conll")),
        "annotator2": sorted((args.input_dir / "annotator2_conll").glob(f"{language}*.conll")),
    }
    
    filenames = {
        "annotator1": [file.name[: file.name.rfind("-")] for file in files["annotator1"]],
        "annotator2": [file.name[: file.name.rfind("-")] for file in files["annotator2"]],
    }
    intersection = set(filenames["annotator1"]) & set(filenames["annotator2"])

    files = {
        "annotator1": sorted([args.input_dir / "labelstudio_conll" / f"{filename}-{annotators[0]}.conll" for filename in intersection]),
        "annotator2": sorted([args.input_dir / "annotator2_conll" / f"{filename}-{annotators[1]}.conll" for filename in intersection]),
    }

    # read tags and compute agreement
    tags = {
        "annotator1": [],
        "annotator2": [],
    }

    for annotator, files in files.items():
        data = []
        for file in files:
            content = read_conll_file(file, TAGSETS["malach"])
            tags[annotator].extend(list(item[1] for item in content))

    f1 = seqeval.metrics.classification_report(tags["annotator1"], tags["annotator2"])
    print(language, f1)