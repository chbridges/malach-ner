from pathlib import Path
import numpy as np
import scipy.stats
import re

def compute_mean_and_confidence(values: list[float]) -> tuple[float, float]:
    n_samples = len(values)

    mean = np.mean(values)
    if n_samples == 1:
        return mean, np.nan
    
    t = scipy.stats.t.ppf(1 - 0.025, n_samples - 1)  # t-score for 0.95 CI
    confidence = t * np.std(values, ddof=1) / np.sqrt(n_samples)

    return mean, confidence

def format_row(key: float, mean: float, confidence: float) -> str:
    tab = "\t" * (3 - len(key) // 4)
    if np.isnan(confidence):
        return f"{key}{tab}{mean * 100:.2f}"
    return f"{key}{tab}{mean * 100:.2f} ± {confidence * 100:.2f}"

def accumulate_experiment(to_accumulate: list[Path]) -> dict[str, str]:
    keys = ["PER", "ORG", "LOC", "CAMP", "GHETTO", "DATE", "MISC", "micro avg"]
    f1_scores = {key: [] for key in keys}

    for file in to_accumulate:
        with file.open() as f:
            lines = [re.sub("PERS", "PER", line.strip()) for line in f.readlines()]
        for line in lines:
            columns = re.split(r"\s\s+", line)
            if len(columns) < 5:
                continue
            key, score = columns[0], float(columns[-2])
            if key in keys:
                f1_scores[key].append(score)

    accumulated = []
    for key, values in f1_scores.items():
        if len(values):
            mean, confidence = compute_mean_and_confidence(values)
            accumulated.append(format_row(key, mean, confidence))

    return "\n".join(accumulated)


if __name__ == "__main__":
    results = Path("./results")

    for directory in results.glob("*"):
        accumulated_file = results / f"{directory.name}_acc.txt"
        accumulated_content = []

        reports = sorted(directory.glob("*.txt"))
        if not reports:
            continue
        unique_experiments = sorted(set([f.name[:f.name.rfind("_")] for f in reports]))

        for handle in unique_experiments:
            to_accumulate = [f for f in reports if handle in f.name]
            acc = accumulate_experiment(to_accumulate)
            accumulated_content.append(f"{handle}\n{acc}")
        
        with accumulated_file.open("w", encoding="utf-8") as f:
            f.write("\n\n".join(accumulated_content))