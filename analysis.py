from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

GENERIC_WORDS = {
    "model", "models", "method", "methods", "result", "results",
    "data", "analysis", "approach", "study", "paper", "using"
}

STOPWORDS = {
    "the", "and", "of", "in", "for", "to", "a", "on", "with", "by",
    "an", "based", "using", "we", "i", "that", "are", "is", "as",
    "this", "from", "which", "be", "at", "or", "it", "these", "their",
    "our", "can", "have", "has"
}

def clean_label(words, top_n=4):
    filtered = []

    for w in words:
        wl = w.lower().strip()

        if wl in STOPWORDS or wl in GENERIC_WORDS:
            continue

        if len(wl) <= 2:
            continue

        if _is_redundant_term(w, filtered):
            continue

        filtered.append(w)

        if len(filtered) >= top_n:
            break

    if not filtered:
        return "Misc Topic"

    return " ".join(filtered).title()


def generate_label(model):
    topic_info = model.get_topic_info().set_index("Topic")
    rows = []

    for topic_id in model.get_topics().keys():
        if topic_id == -1:
            continue

        words = [w for w, _ in model.get_topic(topic_id)]
        label = clean_label(words)

        rows.append({
            "Topic": topic_id,
            "label": label,
            "Count": topic_info.loc[topic_id, "Count"]
        })

    return pd.DataFrame(rows).sort_values("Count", ascending=False)

def assign_quadrant(row):
    if row["slope_24m"] > 0 and row["slope_12m"] > 0:
        return "🚀 Strong & Accelerating"
    elif row["slope_24m"] > 0 and row["slope_12m"] < 0:
        return "📈 Strong but Slowing"
    elif row["slope_24m"] < 0 and row["slope_12m"] > 0:
        return "🌱 Emerging"
    else:
        return "📉 Declining"

def compute_log_trend_scores(trend_share, window=12, min_share=0.001):
    scores = {}

    recent = trend_share.tail(window)
    x = np.arange(len(recent)).reshape(-1, 1)

    mean_share = recent.mean()
    valid_topics = mean_share[mean_share > min_share].index

    for topic in valid_topics:
        y = recent[topic].values
        y_log = np.log(y + 1e-6)

        model = LinearRegression()
        model.fit(x, y_log)

        scores[topic] = model.coef_[0]

    return pd.Series(scores).sort_values(ascending=False)

def choose_time_granularity(df):
    time_span_days = (df["published"].max() - df["published"].min()).days
    n_docs = len(df)

    if time_span_days == 0:
        return "D"

    docs_per_day = n_docs / time_span_days

    if docs_per_day < 1:
        return "M"
    elif docs_per_day < 20:
        return "W"
    else:
        return "D"

def generate_trend_share(df):
    freq = choose_time_granularity(df)

    temp = df.copy()
    temp["time_span"] = temp["published"].dt.to_period(freq).dt.to_timestamp()

    trend = (
        temp[temp["topic"] != -1]
        .groupby(["time_span", "topic"])
        .size()
        .unstack(fill_value=0)
    )

    trend_share = trend.div(trend.sum(axis=1), axis=0)

    return trend_share

def _compute_log_slope(series):
    x = np.arange(len(series)).reshape(-1, 1)
    y = np.asarray(series, dtype=float)

    if np.allclose(y.sum(), 0):
        return None

    y_log = np.log(y + 1e-6)

    model = LinearRegression()
    model.fit(x, y_log)

    return float(model.coef_[0])


def compute_momentum_map_data(trend_share, topic_table, short_window=None, long_window=None):
    """
    Build momentum map data:
    - growth = long-term log slope
    - acceleration = short-term log slope - long-term log slope

    Parameters
    ----------
    trend_share : pd.DataFrame
        index = time bins, columns = topic ids, values = topic share
    topic_table : pd.DataFrame
        must contain ["Topic", "label", "Count"]
    short_window : int | None
        number of last periods for short-term slope
    long_window : int | None
        number of last periods for long-term slope

    Returns
    -------
    pd.DataFrame
        columns:
        - Topic
        - label
        - Count
        - growth
        - short_growth
        - acceleration
        - quadrant
    """

    ts = trend_share.copy()
    n_periods = len(ts)

    if n_periods < 3:
        return pd.DataFrame(columns=[
            "Topic", "label", "Count",
            "growth", "short_growth", "acceleration", "quadrant"
        ])

    # defaults adapt to available timeline
    if long_window is None:
        long_window = n_periods

    if short_window is None:
        short_window = max(3, n_periods // 2)

    long_ts = ts.tail(long_window)
    short_ts = ts.tail(short_window)

    rows = []

    for topic in ts.columns:
        long_slope = _compute_log_slope(long_ts[topic].values)
        short_slope = _compute_log_slope(short_ts[topic].values)

        if long_slope is None or short_slope is None:
            continue

        acceleration = short_slope - long_slope

        rows.append({
            "Topic": topic,
            "growth": long_slope,
            "short_growth": short_slope,
            "acceleration": acceleration
        })

    result = topic_table.merge(pd.DataFrame(rows), on="Topic", how="inner")

    def assign_quadrant(row):
        if row["growth"] > 0 and row["acceleration"] > 0:
            return "Strong & Accelerating"
        elif row["growth"] > 0 and row["acceleration"] < 0:
            return "Growing but Slowing"
        elif row["growth"] < 0 and row["acceleration"] > 0:
            return "Emerging"
        else:
            return "Declining"

    result["quadrant"] = result.apply(assign_quadrant, axis=1)

    return result.sort_values("growth", ascending=False).reset_index(drop=True)


def compute_growth_scores(trend_share, topic_table, window=None):
    """
    Compute growth score using:
        log_slope * log1p(Count)

    Parameters
    ----------
    trend_share : pd.DataFrame
        index = time bins
        columns = topic ids
        values = topic share
    topic_table : pd.DataFrame
        must contain ["Topic", "label", "Count"]
    window : int | None
        if given, use only the last N time points

    Returns
    -------
    pd.DataFrame
        columns:
        - Topic
        - label
        - Count
        - log_slope
        - growth_score
    """

    # Optional rolling window
    if window is not None:
        ts = trend_share.tail(window).copy()
    else:
        ts = trend_share.copy()

    x = np.arange(len(ts)).reshape(-1, 1)
    slopes = []

    for topic in ts.columns:
        y = ts[topic].values.astype(float)

        # Skip empty topics
        if np.allclose(y.sum(), 0):
            continue

        # Log transform for relative growth stability
        y_log = np.log(y + 1e-6)

        model = LinearRegression()
        model.fit(x, y_log)

        slopes.append({
            "Topic": topic,
            "log_slope": float(model.coef_[0])
        })

    slopes_df = pd.DataFrame(slopes)

    result = topic_table.merge(slopes_df, on="Topic", how="inner")

    result["growth_score"] = (
        result["log_slope"] * np.log1p(result["Count"])
    )

    result = result.sort_values("growth_score", ascending=False).reset_index(drop=True)

    return result

def _is_redundant_term(term, selected_terms):
    term_words = set(term.lower().split())

    for prev in selected_terms:
        prev_words = set(prev.lower().split())

        # aynı kelime veya büyük overlap varsa ele
        if term.lower() == prev.lower():
            return True

        if term_words.issubset(prev_words) or prev_words.issubset(term_words):
            return True

    return False

def get_topic_keywords_df(model, topic_id, top_n=8):
    words = model.get_topic(topic_id)

    if not words:
        return None

    cleaned = []
    selected_terms = []

    for w, s in words:
        wl = w.lower().strip()

        if wl in STOPWORDS or wl in GENERIC_WORDS:
            continue

        if len(wl) <= 2:
            continue

        if _is_redundant_term(w, selected_terms):
            continue

        cleaned.append((w, s))
        selected_terms.append(w)

        if len(cleaned) >= top_n:
            break

    if not cleaned:
        return None

    return pd.DataFrame(cleaned, columns=["word", "score"])


def get_representative_docs(df, topic_id, top_n=3):
    topic_docs = df[df["topic"].astype(str) == str(topic_id)].copy()

    if topic_docs.empty:
        return []

    if "text" in topic_docs.columns:
        topic_docs["doc_len"] = topic_docs["text"].astype(str).str.len()
        top_docs = topic_docs.sort_values("doc_len", ascending=False).head(top_n)
    else:
        top_docs = topic_docs.head(top_n)

    results = []

    for _, row in top_docs.iterrows():

        title = str(row.get("title", "Document")).strip()
        pid = str(row.get("paper_id", "")).strip()

        # 🔥 BURASI (senin sorduğun kod)
        if "arxiv.org/abs/" in pid:
            arxiv_id = pid.split("arxiv.org/abs/")[-1]
            url = pid
        else:
            arxiv_id = pid
            url = f"https://arxiv.org/abs/{pid}"

        header = f"[{title}]({url})"
        subtitle = f"arXiv: {arxiv_id}"

        text = str(row.get("text", ""))

        results.append({
            "header": header,
            "subtitle": subtitle,
            "text": text
        })

    return results