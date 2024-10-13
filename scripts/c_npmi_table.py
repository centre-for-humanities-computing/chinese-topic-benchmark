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

summary = data.groupby(["model", "dataset"])["zh_c_npmi"].mean().reset_index()
summary = summary.pivot(index="model", columns="dataset", values="zh_c_npmi")
summary = summary.map(lambda val: f"{val:.2f}")

print(summary.to_latex())
