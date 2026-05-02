import pandas as pd
from data_collect import recursive_download
from embedding_data import get_embeddings
from data_clean import clean_dataset
from topic_model import model_run_pipeline
from analysis import compute_log_trend_scores, assign_quadrant, generate_label
from utils import delta_time
import numpy as np


def run_pipeline(CATEGORIES, start_date, end_date):
    # collect data
    dfs = recursive_download(CATEGORIES, start_date, end_date)
    df = pd.concat(dfs, ignore_index=True)
    
    # prepare data
    df = clean_dataset(df)
    embeddings = get_embeddings(df)
    docs = df['text'].tolist()
    # topic modeling
    model, topics = model_run_pipeline(docs, embeddings)
    df['topics'] = topics
    print(f"Docs: {len(df)}, Topics: {len(set(topics))}")
    return {
        "df": df,
        "docs": docs,
        "embeddings": embeddings,
        "model": model,
        "topics": topics
    }

i, j = delta_time(months="1")
CATEGORIES = ["astro-ph.CO"]
result = run_pipeline(CATEGORIES, i, j)
df = result["df"]
emb = result["embeddings"]
model = result["model"]
topics = result["topics"]
