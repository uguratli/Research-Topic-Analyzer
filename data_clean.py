import pandas as pd
import numpy as np


def clean_dataset(df):
    df = (df.drop_duplicates(subset=['paper_id']).assign(
    published=pd.to_datetime(df["published"]), 
    text= lambda x: x["title"] + " " + x["abstract"])
    .sort_values("published")
    .reset_index(drop=True))
    df = df[['paper_id', "title", 'published', 'text', 'categories']]

    return df
