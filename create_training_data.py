from argparse import ArgumentParser
from pathlib import Path
import json

parser = ArgumentParser(description="Split .conll files into training and test sets based on test_ids.txt.")
parser.add_argument("dataset", type=str, choices=("review", "final"), help="Dataset directory containing 'conll_all' and 'test_ids.txt'.")
args = parser.parse_args()

data_dir = Path(f"./{args.dataset}")

dataset = {"ids": {"train": {}, "test": {}}, "data": {"train": {}, "test": {}}}

with (data_dir / 'test_ids.txt').open() as f:
    test_ids = { line.strip() for line in f }

for file in sorted((data_dir / 'conll_all').iterdir()):
    filename = file.stem
    language = filename[:2]
    
    dataset["ids"]["train"].setdefault(language, [])
    dataset["ids"]["test"].setdefault(language, [])
    dataset["data"]["train"].setdefault(language, [])
    dataset["data"]["test"].setdefault(language, [])

    with file.open() as f:
        data = f.read()
    
    if filename in test_ids:
        dataset["ids"]["test"][language].append(filename)
        dataset["data"]["test"][language].append(data)
    else:
        dataset["ids"]["train"][language].append(filename)
        dataset["data"]["train"][language].append(data)

print(json.dumps(dataset["ids"], indent=2))

dataset_dir = data_dir / "dataset"
dataset_dir.mkdir(exist_ok=True)

for split, data in dataset["data"].items():
    for language, texts in data.items():
        with (dataset_dir / f"{language}.{split}.conll").open("w") as f:
            f.write("\n\n".join(texts))
