import argparse
import json
import re
from pathlib import Path

TAGS = ["PER", "ORG", "LOC", "CAMP", "GHETTO", "DATE", "MISC"]

def parse_args():
    parser = argparse.ArgumentParser(description="Compute tag statistics from annotated data.")
    parser.add_argument("--input_dir", "-i", type=Path, default=Path("./review"), help="Directory containing .conll files.")
    parser.add_argument("--language", "-l", type=str, default=None, help="Optional language filter (e.g., 'en', 'de').")
    parser.add_argument("--force", "-f", action="store_true", help="Force overwrite of existing tag_statistics.json.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    output_file = args.input_dir / "tag_statistics.json"

    if not output_file.exists() or args.force:
        with (args.input_dir / 'annotator_statistics.json').open() as f:
            data = json.load(f)

        annotators = {language: max(annotator, key=annotator.get) for language, annotator in data.items()}

        fulltext = {}
        words = {}
        for language, annotator in annotators.items():
            for file in sorted((args.input_dir / 'labelstudio_conll').glob(f'{language}-*-{annotator}.conll')):
                with open(file, 'r') as f:
                    fulltext[file.name] = f.read()
                with open(file, 'r') as f:
                    words[file.name] = len([line for line in f.readlines() if line.strip()])

        tag_counts = {}
        for filename, content in fulltext.items():
            language = filename[:2]
            tag_counts.setdefault(language, {})
            tag_counts[language][filename] = {}
            for tag in TAGS:
                pattern = f"B-{tag}"
                tag_counts[language][filename][tag] = len(re.findall(pattern, content))
            tag_counts[language][filename]["TOTAL"] = sum(tag_counts[language][filename].values())
            tag_counts[language][filename]["TOKENS"] = words[filename]

        with output_file.open("w") as f:
            json.dump(tag_counts, f, indent=2)
    else:
        with output_file.open() as f:
            tag_counts = json.load(f)

    if args.language:
        print(json.dumps(tag_counts[args.language], indent=2))
    else:
        print(json.dumps(tag_counts, indent=2))
