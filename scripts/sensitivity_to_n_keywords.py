import json
from pathlib import Path

import pandas as pd
from chinese_benchmark import dataset_registry, metric_registry
from tqdm import trange
from turftopic import KeyNMF


def load_keywords(file_path: Path | str) -> list[dict[str, float]]:
    file_path = Path(file_path)
    keywords = []
    with file_path.open() as in_file:
        for line in in_file:
            line = line.strip()
            if line:
                keywords.append(json.loads(line))
    return keywords


CORPORA = ["chinanews", "ihuawen", "oushinet", "xinozhou", "yidali-huarenjie"]
METRICS = ["diversity", "zh_c_npmi", "zh_wec_in", "zh_wec_ex"]

in_dir = Path("dat/keywords")

entries = []
for dataset_name in CORPORA:
    print("Evaluating on ", dataset_name)
    keyword_path = in_dir.joinpath(f"{dataset_name}_all_gen_keywords.jsonl")
    keywords = load_keywords(keyword_path)
    corpus = dataset_registry.get(dataset_name)()
    for top_n in trange(5, 105, 5, desc="Evaluating on different Top N"):
        model = KeyNMF(10, top_n=top_n)
        doc_topic_matrix = model.fit(corpus, keywords=keywords)
        topic_data = dict(
            corpus=corpus,
            vocab=model.get_vocab(),
            document_topic_matrix=doc_topic_matrix,
            topic_term_matrix=model.components_,
            topic_names=model.topic_names,
        )
        res = {"top_n": top_n, "dataset_name": dataset_name}
        for metric_name, metric_loader in metric_registry.get_all().items():
            if metric_name in METRICS:
                metric = metric_loader()
                score = metric(topic_data, dataset_name)
                res[metric_name] = float(score)
        entries.append(res)

df = pd.DataFrame.from_records(entries)
df.to_csv("dat/keyword_sensitivity.csv")
