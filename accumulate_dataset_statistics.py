import json
from pathlib import Path

data = Path("./final")
with (data / "test_ids.txt").open() as f:
    test_ids = set(line.strip() for line in f.readlines())
with (data / "tag_statistics.json").open() as f:
    statistics = json.load(f)

accumulated = {
    language: {
        split: {
            "TOKENS": 0,
            "PER": 0,
            "ORG": 0,
            "LOC": 0,
            "CAMP": 0,
            "GHETTO": 0,
            "DATE": 0,
            "MISC": 0,
            "TOTAL": 0,
        } for split in ["train", "test", "total"]
    } for language in list(statistics.keys()) + ["total"]
}
for language, files in statistics.items():
    for file, stats in files.items():
        split = "test" if file[:-6] in test_ids else "train"
        for tag, count in stats.items():
            accumulated[language][split][tag] += count
            accumulated[language]["total"][tag] += count
            accumulated["total"][split][tag] += count
            accumulated["total"]["total"][tag] += count

with (data / "dataset_statistics.json").open("w") as f:
    json.dump(accumulated, f, indent=2)