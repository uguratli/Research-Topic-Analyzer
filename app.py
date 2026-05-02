import streamlit as st
from datetime import date, timedelta
import pandas as pd
import plotly.express as px

from data_collect import recursive_download
from embedding_data import get_embeddings
from data_clean import clean_dataset
from topic_model import model_run_pipeline
from analysis import *
import json

@st.cache_data
def load_categories():
    with open("arxiv_categories.json", "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)

categories_df = load_categories()

categories_df["display_name"] = (
    categories_df["category_id"] + " — " + categories_df["category_name"]
)


def interpret_topic(growth_score, share_change):
    if growth_score > 0.01 and share_change > 0:
        return "🚀 Rapidly Growing"
    elif growth_score > 0:
        return "📈 Growing"
    elif growth_score < 0 and share_change < 0:
        return "📉 Declining"
    else:
        return "⚖️ Stable"

st.set_page_config(page_title="Topic Finder", layout="wide")


# -----------------------------
# SESSION STATE
# -----------------------------
if "running" not in st.session_state:
    st.session_state.running = False

if "result" not in st.session_state:
    st.session_state.result = None

# -----------------------------
# PAGE HEADER
# -----------------------------
st.title("🔬 Research Topic Analyzer")
st.caption(
    "Explore emerging research topics by category and date range using topic modeling and trend analysis."
)

categories = [
    "astro-ph.CO",
    "astro-ph.GA",
    "cs.AI",
    "cs.CL",
    "cs.LG",
    "stat.ML"
]

today = date.today()
one_year_ago = today - timedelta(days=365)

st.subheader("Select Inputs")

col1, col2 = st.columns([2, 1])

with col1:
    selected_dates = st.slider(
        "Select Date Range",
        min_value=one_year_ago,
        max_value=today,
        value=(today - timedelta(days=30), today),
        format="YYYY-MM-DD",
        disabled=st.session_state.running
    )

with col2:
    selected_display = st.multiselect(
        "Select Categories",
        options=categories_df["display_name"].tolist(),
        default=[categories_df["display_name"].iloc[0]],
        disabled=st.session_state.running
    )

    selected_categories = (
        categories_df[categories_df["display_name"].isin(selected_display)]["category_id"]
        .tolist()
    )
    if selected_display:

        selected_info = categories_df[
            categories_df["display_name"].isin(selected_display)
        ][["category_id", "category_name", "description"]]

        with st.expander("Category descriptions"):
            st.dataframe(selected_info, use_container_width=True)

analyze_clicked = st.button(
    "Analyze",
    disabled=st.session_state.running
)

# -----------------------------
# TRIGGER ANALYSIS
# -----------------------------
if analyze_clicked:

    # 🔥 CATEGORY LIMIT CHECK
    if len(selected_categories) == 0:
        st.warning("Please select at least one category.")
        st.stop()

    if len(selected_categories) > 3:
        st.warning("Please select up to 3 categories for better results.")
        st.stop()

    # OK → run
    st.session_state.running = True
    st.session_state.result = None
    st.rerun()

# -----------------------------
# RUN ANALYSIS
# -----------------------------
if st.session_state.running:
    progress = st.progress(0)
    status_text = st.empty()

    try:
        start_date, end_date = selected_dates

        # Step 1 — Data collection
        status_text.info("Collecting papers...")
        progress.progress(20)

        dfs = recursive_download(selected_categories, start_date, end_date)
        df = pd.concat(dfs, ignore_index=True)

        # Step 2 — Cleaning
        status_text.info("Cleaning dataset...")
        progress.progress(40)
        df = df.sort_values(["published", "paper_id"]).reset_index(drop=True)
        df = clean_dataset(df)

        # Step 3 — Embeddings
        status_text.info("Generating embeddings...")
        progress.progress(60)

        embeddings = get_embeddings(df)

        # Step 4 — Topic modeling
        status_text.info("Running topic modeling...")
        progress.progress(80)

        docs = df["text"].tolist()
        model, topics = model_run_pipeline(docs, embeddings)
        df["topic"] = topics

        # Step 5 — Analysis preparation
        status_text.info("Preparing analysis outputs...")
        progress.progress(95)

        topic_labels_df = generate_label(model)
        trend_share = generate_trend_share(df)

        # Merge label info into df if needed later
        df = df.merge(
            topic_labels_df[["Topic", "label", "Count"]],
            left_on="topic",
            right_on="Topic",
            how="left"
        )

        preview_df = df[["published", "topic", "label"]].head(5).copy()

        st.session_state.result = {
            "categories": selected_categories,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "num_papers": len(df),
            "num_topics": len(set(t for t in topics if t != -1)),
            "num_outliers": int((df["topic"] == -1).sum()),
            "preview_df": preview_df,
            "topic_labels_df": topic_labels_df,
            "trend_share": trend_share,
            "df": df,
            "model": model
        }

        progress.progress(100)
        status_text.success("Analysis completed successfully.")

    except Exception as e:
        st.error(f"Analysis failed: {e}")

    finally:
        st.session_state.running = False
        st.rerun()

# -----------------------------
# SHOW RESULTS
# -----------------------------
if st.session_state.result is not None:
    result = st.session_state.result
    model = result["model"]
    analysis_df = result["df"]

    st.success("✅ Analysis completed")

    # Summary
    st.header("1. Analysis Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Categories", len(result["categories"]))
    col2.metric("Documents", f"{result['num_papers']:,}")
    col3.metric("Topics", result["num_topics"])
    col4.metric("Outliers", result["num_outliers"])

    st.caption(
        f"Date Range: {result['start_date']} → {result['end_date']}"
    )

    st.divider()

    # Topic overview table
    st.header("2. Topic Overview")

    topic_table = result["topic_labels_df"][["Topic", "label", "Count"]].sort_values(
        "Count", ascending=False
    )
    growth_df = compute_growth_scores(
        trend_share=result["trend_share"],
        topic_table=topic_table,
        window=None
    )

    st.dataframe(topic_table, use_container_width=True)

    st.divider()

    # Topic count bar chart
    st.subheader("Topic Size Distribution")

    fig_topic_sizes = px.bar(
        topic_table.sort_values("Count", ascending=False).head(15),
        x="label",
        y="Count",
        title="Top Topics by Number of Documents"
    )

    fig_topic_sizes.update_layout(
        template="plotly_white",
        xaxis_title="Topic",
        yaxis_title="Document Count",
        xaxis_tickangle=-30
    )

    st.plotly_chart(fig_topic_sizes, use_container_width=True)

    st.caption(
        "This chart shows the largest discovered topics based on document count."
    )

    st.divider()

        # -----------------------------
    # TOP TOPICS OVER TIME
    # -----------------------------
    st.subheader("Top Topics Over Time")

    trend_share = result["trend_share"].copy()

    top_topic_ids = topic_table.head(5)["Topic"].tolist()
    available_topic_ids = [t for t in top_topic_ids if t in trend_share.columns]

    if available_topic_ids:
        plot_df = trend_share[available_topic_ids].reset_index()

        if "time_span" not in plot_df.columns:
            plot_df = plot_df.rename(columns={plot_df.columns[0]: "time_span"})

        topic_label_map = topic_table.set_index("Topic")["label"].to_dict()
        rename_map = {t: topic_label_map.get(t, str(t)) for t in available_topic_ids}

        plot_df = plot_df.rename(columns=rename_map)

        fig_over_time = px.line(
            plot_df,
            x="time_span",
            y=list(rename_map.values()),
            title="Top Topics Over Time"
        )

        fig_over_time.update_layout(
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="Topic Share",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=50, b=10)
        )

        st.plotly_chart(fig_over_time, use_container_width=True)

        st.caption(
            "This chart shows how the largest topics change over time based on their relative share."
        )

    else:
        st.info("Not enough topic trend data to show topics over time.")

    st.divider()

    st.header("3. Trend Intelligence")
    # -----------------------------
    # TOP GROWING TOPICS
    # -----------------------------
    st.subheader("Top Growing Topics")

    top_growing = (
        growth_df[growth_df["growth_score"] > 0]
        .sort_values("growth_score", ascending=False)
        .head(10)
    )

    if len(top_growing) > 0:
        st.dataframe(
            top_growing[["label", "Count", "log_slope", "growth_score"]],
            use_container_width=True
        )

        st.caption(
            "Growth score combines relative growth (log slope) and topic size. "
            "Topics that are both growing and substantial rank higher."
        )

        fig_growth = px.bar(
            top_growing,
            x="label",
            y="growth_score",
            title="Top Growing Topics"
        )

        fig_growth.update_layout(
            template="plotly_white",
            xaxis_title="Topic",
            yaxis_title="Growth Score",
            xaxis_tickangle=-30
        )

        st.plotly_chart(fig_growth, use_container_width=True)

    else:
        st.info("No strongly growing topics detected in this time range.")

    st.divider()

    # -----------------------------
    # TOP DECLINING TOPICS
    # -----------------------------
    st.subheader("Top Declining Topics")

    top_declining = (
        growth_df[growth_df["growth_score"] < 0]
        .sort_values("growth_score", ascending=True)
        .head(10)
    )

    if len(top_declining) > 0:
        st.dataframe(
            top_declining[["label", "Count", "log_slope", "growth_score"]],
            use_container_width=True
        )

        st.caption(
            "These topics are declining in relative share over time."
        )

        fig_decline = px.bar(
            top_declining,
            x="label",
            y="growth_score",
            title="Top Declining Topics"
        )

        fig_decline.update_layout(
            template="plotly_white",
            xaxis_title="Topic",
            yaxis_title="Growth Score",
            xaxis_tickangle=-30
        )

        st.plotly_chart(fig_decline, use_container_width=True)

    else:
        st.info("No strongly declining topics detected in this time range.")

    st.divider()
    # -----------------------------
    # TOP IMPACT TOPICS
    # -----------------------------
    st.subheader("Top Impact Topics")

    impact_df = (
        growth_df[growth_df["growth_score"] > 0]
        .sort_values("growth_score", ascending=False)
        .head(10)
    )

    if len(impact_df) > 0:
        st.dataframe(
            impact_df[["label", "Count", "log_slope", "growth_score"]],
            use_container_width=True
        )

        st.caption(
            "Impact highlights topics that are both growing and substantial in size."
        )

        fig_impact = px.bar(
            impact_df,
            x="label",
            y="growth_score",
            title="Top Impact Topics"
        )

        fig_impact.update_layout(
            template="plotly_white",
            xaxis_title="Topic",
            yaxis_title="Impact Score",
            xaxis_tickangle=-30
        )

        st.plotly_chart(fig_impact, use_container_width=True)

    else:
        st.info("No high-impact topics detected in this time range.")

    st.divider()
    # -----------------------------
    # MOMENTUM MAP
    # -----------------------------
    st.header("4. Momentum Map")

    momentum_df = compute_momentum_map_data(
        trend_share=result["trend_share"],
        topic_table=topic_table
    )

    if len(momentum_df) > 0:
        fig_momentum = px.scatter(
            momentum_df,
            x="acceleration",
            y="growth",
            size="Count",
            color="quadrant",
            hover_name="label",
            hover_data={
                "Count": True,
                "growth": ':.4f',
                "short_growth": ':.4f',
                "acceleration": ':.4f'
            },
            title="Topic Momentum Map"
        )

        fig_momentum.add_vline(x=0, line_dash="dash")
        fig_momentum.add_hline(y=0, line_dash="dash")

        fig_momentum.update_layout(
            template="plotly_white",
            xaxis_title="Acceleration (Recent vs Overall)",
            yaxis_title="Growth Trend"
        )

        st.plotly_chart(fig_momentum, use_container_width=True)

        st.caption(
            "This map compares long-term growth and short-term acceleration. "
            "Upper-right topics are strong and accelerating, while lower-left topics are declining."
        )

    else:
        st.info("Not enough trend data to build a momentum map.")
    st.caption(
    "Use this map to identify interesting topics, then select them below for detailed analysis."
    )

    # -----------------------------
    # TOPIC COMPARISON
    # -----------------------------
    st.header("5. Topic Explorer")

    trend_share = result["trend_share"].copy()

    topic_options = topic_table["Topic"].tolist()
    topic_label_map = topic_table.set_index("Topic")["label"].to_dict()

    default_topics = top_growing["Topic"].head(3).tolist()

    selected_topics = st.multiselect(
        "Select topics to compare",
        options=topic_options,
        default=default_topics,
        format_func=lambda x: topic_label_map.get(x, str(x))
    )

    if selected_topics:
        available_topics = [t for t in selected_topics if t in trend_share.columns]

        if available_topics:
            plot_df = trend_share[available_topics].reset_index()

            # zaman kolonu adı garanti
            if "time_span" not in plot_df.columns:
                plot_df = plot_df.rename(columns={plot_df.columns[0]: "time_span"})

            # topic id -> label
            rename_map = {t: topic_label_map.get(t, str(t)) for t in available_topics}
            plot_df = plot_df.rename(columns=rename_map)

            fig_compare = px.line(
                plot_df,
                x="time_span",
                y=list(rename_map.values()),
                title="Topic Trends Comparison"
            )

            fig_compare.update_layout(
                template="plotly_white",
                xaxis_title="Time",
                yaxis_title="Topic Share",
                hovermode="x unified"
            )

            st.plotly_chart(fig_compare, use_container_width=True)

            st.caption(
                "This chart compares the relative share of selected topics over time."
            )
        else:
            st.warning("Selected topics were not found in trend data.")
    else:
        st.info("Select at least one topic to compare.")

    st.divider()
    # -----------------------------
    # DETAILED TOPIC ANALYSIS
    # -----------------------------
    st.header("6. Detailed Topic Analysis")

    if selected_topics:

        for topic_id in selected_topics:

            if topic_id not in trend_share.columns:
                continue

            topic_label = topic_label_map.get(topic_id, str(topic_id))

            with st.container(border=True):

                st.markdown(f"### {topic_label}")

                topic_row = topic_table[topic_table["Topic"] == topic_id].iloc[0]

                g_row = growth_df[growth_df["Topic"] == topic_id]
                m_row = momentum_df[momentum_df["Topic"] == topic_id]

                growth_score = g_row["growth_score"].values[0] if not g_row.empty else 0
                log_slope = g_row["log_slope"].values[0] if not g_row.empty else 0
                quadrant = m_row["quadrant"].values[0] if not m_row.empty else "Unknown"
                latest_share = trend_share[topic_id].iloc[-1] if topic_id in trend_share.columns else 0
                # -----------------
                # SHARE CHANGE
                # -----------------
                series = trend_share[topic_id]

                start_share = series.iloc[0]
                end_share = series.iloc[-1]

                share_change = (end_share - start_share) 
                badge = interpret_topic(growth_score, share_change)

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Docs", int(topic_row["Count"]))
                col2.metric("Growth", f"{growth_score:.3f}")
                col3.metric("Trend", quadrant)
                col4.metric(
                    "Share",
                    f"{latest_share:.2%}",
                    delta=f"{share_change:+.2%}"
                )
                
                plot_df = trend_share[[topic_id]].reset_index()
                st.markdown(f"**Status:** {badge}")

                
                if "time_span" not in plot_df.columns:
                    plot_df = plot_df.rename(columns={plot_df.columns[0]: "time_span"})

                plot_df = plot_df.rename(columns={topic_id: topic_label})

                fig_topic = px.line(
                    plot_df,
                    x="time_span",
                    y=topic_label,
                    title=f"Trend of {topic_label}"
                )

                fig_topic.update_layout(
                    template="plotly_white",
                    xaxis_title="Time",
                    yaxis_title="Topic Share"
                )

                st.plotly_chart(fig_topic, use_container_width=True)

                kw_df = get_topic_keywords_df(model, topic_id)

                if kw_df is not None:
                    st.markdown("**Top Keywords**")

                    fig_kw = px.bar(
                        kw_df.sort_values("score"),
                        x="score",
                        y="word",
                        orientation="h",
                        title=None
                    )

                    fig_kw.update_layout(
                        template="plotly_white",
                        xaxis_title="Importance",
                        yaxis_title=""
                    )

                    st.plotly_chart(fig_kw, use_container_width=True)
                else:
                    st.info("No keywords available")

                papers = get_representative_docs(analysis_df, topic_id)

                if papers:
                    st.markdown("**Representative Papers**")

                    for p in papers:
                        st.markdown(f"🔗 {p['header']}")
                        st.caption(p.get("subtitle", ""))

                        with st.expander("Show full text"):
                            st.write(p["text"])
                else:
                    st.info("No representative papers found")
                st.markdown(" ")
