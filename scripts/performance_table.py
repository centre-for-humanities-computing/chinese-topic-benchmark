import json
from pathlib import Path
from typing import Optional

import pandas as pd


def safe_index(elements: list, elem) -> Optional[int]:
    try:
        return elements.index(elem)
    except ValueError:
        return None


records = []
with Path("results/paraphrase-multilingual-MiniLM-L12-v2.jsonl").open() as in_file:
    for line in in_file:
        line = line.strip()
        if line:
            entry = json.loads(line)
            res = entry.pop("results")
            entry = {**entry, **res}
            records.append(entry)

data = pd.DataFrame.from_records(records)

METRICS = ["zh_c_npmi", "diversity", "zh_wec_in", "zh_wec_ex"]
FORMATTED_METRICS = [
    "$C_{\\text{NPMI}}$",
    "$d$",
    "$C_{\\text{in}}$",
    "$C_{\\text{ex}}$",
]
MODELS = [
    "KeyNMF",
    "S³",
    "Top2Vec",
    "BERTopic",
    "CombinedTM",
    "ZeroShotTM",
    "NMF",
    "LDA",
]
DATASETS = [
    "chinanews",
    "ihuawen",
    "oushinet",
    "xinozhou",
    "yidali-huarenjie",
]
summary = data.groupby(["dataset", "model"])[METRICS].mean().reset_index()
lines = []
n_datasets = len(DATASETS)
top_table_layout = "c" * (n_datasets + 1)
lines.extend(
    [
        f"\\begin{{tabular}}{{{top_table_layout}}}",
        "\\toprule",
        "&" + " & ".join([f"\\textbf{{{dataset}}}" for dataset in DATASETS]) + "\\\\",
        "\n",
    ]
)
lines.extend(
    [
        "\\begin{tabular}{l}",
        "\\textbf{Model} \\\\",
        *[f"\\textbf{{{model}}} \\\\" for model in MODELS],
        "\\end{tabular} &",
    ]
)
for dataset in DATASETS:
    res = summary[summary["dataset"] == dataset]
    subtable_layout = "c" * len(METRICS)
    lines.extend(
        [
            f"\\begin{{tabular}}{{{subtable_layout}}}",
            " & ".join(FORMATTED_METRICS) + "\\\\",
            "\\midrule",
        ]
    )
    res = res[res["model"].isin(MODELS)]
    res = res.set_index("model")
    best_models = {metric: list(res[metric].nlargest(2).index) for metric in METRICS}
    for model in MODELS:
        if model not in res.index:
            continue
        entry = res.loc[model]
        formatted_metrics = []
        for metric in METRICS:
            metric_result = entry[metric]
            metric_formatted = f"{metric_result:.2f}"
            model_rank = safe_index(best_models[metric], model)
            if model_rank == 0:
                metric_formatted = f"\\textbf{{{metric_formatted}}}"
            if model_rank == 1:
                metric_formatted = f"\\underline{{{metric_formatted}}}"
            formatted_metrics.append(metric_formatted)
        lines.append(" & ".join(formatted_metrics) + "\\\\")
    lines.extend(
        [
            "\\end{tabular} &",
            "\n",
        ]
    )
lines.append("\\end{tabular}")
lines.append("\n")
print("\n".join(lines))
