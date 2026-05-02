from sentence_transformers import SentenceTransformer


def get_embeddings(df, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True, batch_size=64)
    return embeddings