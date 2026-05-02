from bertopic import BERTopic
from sklearn.cluster import KMeans
from umap import UMAP
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer

random.seed(42)
np.random.seed(42)

CUSTOM_STOPWORDS = {
    "using", "based", "results", "result", "study", "analysis",
    "method", "methods", "approach", "approaches",
    "model", "models", "data", "paper"
}


def build_vectorizer():
    return CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.8
    )

def build_umap():
    return UMAP(
        n_neighbors=15,
        n_components=5,
        metric="cosine",
        random_state=42
    )


def get_target_topics(n_docs):
    if n_docs < 500:
        return 10
    elif n_docs < 2000:
        return 20
    elif n_docs < 10000:
        return 35
    else:
        return 50

def model_run_pipeline(docs, embeddings):
    n_docs = len(docs)
    target_topics = get_target_topics(n_docs)

    umap_model = build_umap()
    reduced_embeddings = umap_model.fit_transform(embeddings)

    cluster_model = KMeans(
        n_clusters=target_topics,
        random_state=42,
        n_init=10
    )

    model = BERTopic(
        umap_model=None,
        hdbscan_model=cluster_model,
        vectorizer_model=build_vectorizer(),   # 🔥 EKLE
        calculate_probabilities=False
    )

    topics, _ = model.fit_transform(docs, reduced_embeddings) # type: ignore

    return model, topics