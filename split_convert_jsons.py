import json
import os
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("dataset", choices=("review", "final"))
parser.add_argument("--split", "-s", action="store_true")
parser.add_argument("--convert", "-c", action="store_true")
args = parser.parse_args()

split_dir = Path(f"./{args.dataset}/json_split/")
conll_dir = Path(f"./{args.dataset}/conll_all/")
stats_file = Path(f"./{args.dataset}/annotator_statistics.json")

if args.split:
    split_dir.mkdir(exist_ok=True)
    annotator_statistics = {}

    for filepath in tqdm(sorted(Path(f"./{args.dataset}/json_raw/").glob("*.json")), desc="Splitting"):
        language = filepath.stem
        with filepath.open(encoding="utf-8") as file:
            export = json.load(file)

        for tape in export:
            split = {"id": tape["id"], "data": tape["data"]}
            meta = "-".join(tape["data"]["meta"].values())
            for annotation in tape["annotations"]:
                annotator = annotation["completed_by"]
                split["annotations"] = [{
                    "result": annotation["result"],
                    "completed_by": annotator,
                }]
                handle = f"{meta}-{annotator}.json"
                annotator_statistics.setdefault(language, {}).setdefault(annotator, 0)
                annotator_statistics[language][annotator] += 1

                with (split_dir / handle).open("w", encoding="utf-8") as file:
                    json.dump([split], file, ensure_ascii=False, indent=2)

    with stats_file.open("w") as file:
        json.dump(annotator_statistics, file, indent=2)

if args.convert:
    conll_dir.mkdir(exist_ok=True)
    with stats_file.open() as file:
        annotator_statistics = json.load(file)

    for language, annotators in annotator_statistics.items():
        annotator = max(annotators, key=annotators.get)

        for filepath in tqdm(sorted(split_dir.glob(f"{language}-*-{annotator}.json")), desc=f"Converting {language}"):
            infile = str(filepath.absolute())
            os.system(f"cd ../label-studio-converter; bash convert.sh {infile}")

            converted = filepath.with_suffix(".conll")
            (converted / "result.conll").rename(conll_dir / converted.name)
            converted.rmdir()
