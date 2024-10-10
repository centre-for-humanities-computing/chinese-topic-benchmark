import itertools
import json
import os
from pathlib import Path
from typing import Optional

import jieba
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from gensim.models.coherencemodel import CoherenceModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import paired_cosine_distances
from topic_benchmark.base import Metric
from topic_benchmark.benchmark import run_benchmark
from topic_benchmark.cli import load_cache
from topic_benchmark.metrics.wec import word_embedding_coherence
from topic_benchmark.registries import dataset_registry, metric_registry
from topic_benchmark.utils import get_top_k
from turftopic.data import TopicData

os.environ["TOKENIZERS_PARALLELISM"] = "false"

encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def tokenize_zh(text):
    return jieba.lcut(text)


with open("dat/chinese_stopwords.txt") as file:
    stopwords = file.readlines()
stopwords = [s.strip() for s in stopwords]
vectorizer = CountVectorizer(stop_words=stopwords, tokenizer=tokenize_zh)


def external_coherence(topics, embedding_model: SentenceTransformer):
    arrays = []
    for index, topic in enumerate(topics):
        if len(topic) > 0:
            embeddings = dict(zip(topic, embedding_model.encode(topic)))
            w1, w2 = zip(*itertools.combinations(topic, 2))
            e1 = np.stack([embeddings[w] for w in w1])
            e2 = np.stack([embeddings[w] for w in w2])
            similarities = 1 - paired_cosine_distances(e1, e2)
            arrays.append(np.nanmean(similarities))
    return np.nanmean(arrays)


@metric_registry.register("zh_wec_ex")
def load_embedding_coherence_ex() -> Metric:
    """Multilingual external embedding coherence with a
    multilingual sentence transformer instead of a word embedding model
    """
    top_k = 10

    def score(data: TopicData, dataset_name: Optional[str]):
        topics = get_top_k(data, top_k)
        return external_coherence(topics, encoder)

    return score


@metric_registry.register("zh_c_npmi")
def load_npmi() -> Metric:
    top_k = 10

    def score(data: TopicData, dataset_name: Optional[str]):
        topics = get_top_k(data, top_k)
        tokenizer = CountVectorizer(
            vocabulary=data["vocab"], tokenizer=tokenize_zh, stop_words=stopwords
        ).build_analyzer()
        texts = [tokenizer(text) for text in data["corpus"]]
        dictionary = Dictionary(texts)
        cm = CoherenceModel(
            topics=topics,
            texts=texts,
            dictionary=dictionary,
            coherence="c_npmi",
        )
        return cm.get_coherence()

    return score


@metric_registry.register("zh_wec_in")
def load_iwec() -> Metric:
    top_k = 10

    # Cache for w2v models over corpora
    w2v_cache: dict[str, Word2Vec] = {}

    def score(data: TopicData, dataset_name: Optional[str]):
        if dataset_name not in w2v_cache:
            tokenizer = CountVectorizer(
                vocabulary=data["vocab"], tokenizer=tokenize_zh, stop_words=stopwords
            ).build_analyzer()
            texts = [tokenizer(text) for text in data["corpus"]]
            model = Word2Vec(texts, min_count=1)
            w2v_cache[dataset_name] = model
        else:
            model = w2v_cache[dataset_name]
        topics = get_top_k(data, top_k)
        return word_embedding_coherence(topics, model.wv)

    return score


@dataset_registry.register("chinanews")
def load_chinanews():
    corpus = pd.read_csv("dat/chinanews_all.csv")["text"].dropna()
    return list(corpus)


@dataset_registry.register("ihuawen")
def load_ihuawen():
    corpus = pd.read_csv("dat/ihuawen_all.csv")["text"].dropna()
    return list(corpus)


@dataset_registry.register("oushinet")
def load_oushinet():
    corpus = pd.read_csv("dat/oushinet_all.csv")["text"].dropna()
    return list(corpus)


@dataset_registry.register("xinozhou")
def load_xinozhou():
    corpus = pd.read_csv("dat/xinozhou_all.csv")["text"].dropna()
    return list(corpus)


@dataset_registry.register("yidali-huarenjie")
def load_yidali_huarenjie():
    corpus = pd.read_csv("dat/yidali-huarenjie_all.csv")["text"].dropna()
    return list(corpus)


CORPORA = ["chinanews", "ihuawen", "oushinet", "xinozhou", "yidali-huarenjie"]


def main():
    encoder_name = "paraphrase-multilingual-MiniLM-L12-v2"
    encoder_path_name = encoder_name.replace("/", "__")
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir.joinpath(f"{encoder_path_name}.jsonl")
    out_path = f"results/{encoder_path_name}.jsonl"
    cached_entries = load_cache(out_path)
    print("--------------------------------------")
    print(f"Running benchmark with {encoder_name}")
    print("--------------------------------------")
    entries = run_benchmark(
        encoder=encoder,
        vectorizer=vectorizer,
        models=[
            "BERTopic",
            "NMF",
            "LDA",
            "Top2Vec",
            "KeyNMF",
            "S³",
            "CombinedTM",
            "ZeroShotTM",
        ],
        datasets=CORPORA,
        metrics=["diversity", "zh_c_npmi", "zh_wec_in", "zh_wec_ex"],
        seeds=(42,),
        prev_entries=cached_entries,
    )
    for entry in entries:
        with open(out_path, "a") as out_file:
            out_file.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
