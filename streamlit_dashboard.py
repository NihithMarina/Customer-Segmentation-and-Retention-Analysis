from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


st.set_page_config(
    page_title="RetentionPro - Customer Retention Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


BASE_DIR = Path(__file__).resolve().parent
FILES = {
    "customer_segments": BASE_DIR / "customer_segments.csv",
    "segment_summary": BASE_DIR / "segment_summary.csv",
    "clean_transactions": BASE_DIR / "clean_transactions.csv",
    "churn_predictions": BASE_DIR / "churn_predictions.csv",
}


sns.set_theme(style="whitegrid")


def infer_first_existing(columns: pd.Index, candidates: list[str]) -> Optional[str]:
    cols = set(columns)
    for item in candidates:
        if item in cols:
            return item
    return None


def normalize_customer_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(".0", "", regex=False).str.strip()


@st.cache_data(show_spinner=False)
def load_data() -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}

    for name, path in FILES.items():
        if not path.exists():
            continue
        data[name] = pd.read_csv(path)

    if "customer_segments" in data:
        cs = data["customer_segments"].copy()
        for col in ["last_purchase_date", "first_purchase_date"]:
            if col in cs.columns:
                cs[col] = pd.to_datetime(cs[col], errors="coerce")

        numeric_cols = [
            "monetary",
            "frequency",
            "recency",
            "tenure_days",
            "avg_order_value",
            "cluster",
            "churn",
            "churn_probability",
        ]
        for col in numeric_cols:
            if col in cs.columns:
                cs[col] = pd.to_numeric(cs[col], errors="coerce")

        if "customer_id" in cs.columns:
            cs["customer_id"] = normalize_customer_id(cs["customer_id"])

        if "segment_name" not in cs.columns and "cluster" in cs.columns:
            cs["segment_name"] = "Cluster " + cs["cluster"].astype("Int64").astype(str)

        if "churn_probability" in cs.columns:
            cs["churn_probability"] = cs["churn_probability"].clip(0.0, 1.0)

        data["customer_segments"] = cs

    if "clean_transactions" in data:
        tx = data["clean_transactions"].copy()
        for date_col in ["invoicedate", "invoice_date", "order_date", "date"]:
            if date_col in tx.columns:
                tx[date_col] = pd.to_datetime(tx[date_col], errors="coerce")

        for ncol in ["order_amount", "quantity", "unitprice", "unit_price"]:
            if ncol in tx.columns:
                tx[ncol] = pd.to_numeric(tx[ncol], errors="coerce")

        data["clean_transactions"] = tx

    return data


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --dark-bg: #0d1117;
            --dark-secondary: #161b22;
            --dark-tertiary: #21262d;
            --accent-cyan: #00bcd4;
            --accent-purple: #7c3aed;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --border-color: #30363d;
        }

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
            background-color: var(--dark-bg) !important;
            color: var(--text-primary);
        }

        h1, h2, h3, h4, h5 {
            color: var(--text-primary) !important;
            font-weight: 700;
        }

        h1 { font-size: 2.3rem !important; }
        h2 { font-size: 1.6rem !important; }
        h3 { font-size: 1.3rem !important; }

        [data-testid="stAppViewContainer"] {
            background-color: var(--dark-bg) !important;
        }

        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] span,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] li,
        [data-testid="stAppViewContainer"] div {
            color: var(--text-secondary);
        }

        [data-testid="stSidebar"] {
            background-color: var(--dark-secondary) !important;
            border-right: 1px solid var(--border-color) !important;
        }

        [data-testid="stSidebar"] * {
            color: var(--text-primary) !important;
        }

        /* Hide Streamlit sidebar; we render controls in the page layout */
        [data-testid="stSidebar"] {
            display: none !important;
        }

        /* Page container alignment */
        [data-testid="stAppViewContainer"] .block-container {
            max-width: 1400px;
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            margin: 0 auto;
        }

        /* Horizontal nav (radio) alignment */
        div[role="radiogroup"] {
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            align-items: center;
        }

        div[role="radiogroup"] > label {
            margin-right: 0 !important;
        }

        [data-testid="stSidebar"] [role="button"] {
            background-color: var(--dark-tertiary) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--accent-cyan) !important;
            font-weight: 600 !important;
        }

        [data-testid="stSidebar"] [role="button"]:hover {
            background-color: var(--accent-cyan) !important;
            color: var(--dark-bg) !important;
        }

        .hero {
            background-color: var(--dark-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 2.5rem;
            margin-bottom: 2rem;
        }

        .hero h1 {
            margin: 0;
            color: var(--accent-cyan) !important;
            font-size: 2.8rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }

        .hero .subtitle {
            color: var(--accent-purple);
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 1rem;
            letter-spacing: 1px;
            text-transform: uppercase;
        }

        .hero p {
            margin: 0;
            color: var(--text-primary);
            max-width: 900px;
            font-size: 1rem;
            line-height: 1.6;
            font-weight: 400;
        }

        button[role="tab"] {
            font-weight: 600 !important;
            color: var(--text-secondary) !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
            background: transparent !important;
            font-size: 1rem !important;
        }

        button[role="tab"][aria-selected="true"] {
            color: var(--accent-cyan) !important;
            border-bottom: 2px solid var(--accent-cyan) !important;
        }

        button[role="tab"]:hover {
            color: var(--accent-cyan) !important;
        }

        [data-testid="metric-container"] {
            background-color: var(--dark-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1.2rem;
        }

        [data-testid="metric-container"] label,
        [data-testid="metric-container"] div,
        [data-testid="metric-container"] p,
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricValue"] {
            color: var(--text-primary) !important;
        }

        [data-testid="stMetricDelta"] {
            color: #10b981 !important;
        }

        .kpi-card {
            background-color: var(--dark-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1.5rem;
            min-height: 130px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: all 0.2s ease;
        }

        .kpi-card:hover {
            background-color: #1a2332;
            border-color: var(--accent-cyan);
        }

        .kpi-label {
            color: var(--accent-cyan);
            font-size: 0.75rem;
            margin-bottom: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 700;
        }

        .kpi-value {
            font-size: 1.9rem;
            font-weight: 800;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }

        .kpi-note {
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: auto;
        }

        .insight-card {
            background-color: var(--dark-tertiary);
            border: 1px solid var(--border-color);
            border-left: 4px solid var(--accent-purple);
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
        }

        .insight-card strong {
            color: var(--accent-cyan);
            font-weight: 700;
        }

        .section-header {
            color: var(--text-primary);
            font-weight: 700;
            font-size: 1.4rem;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def fmt_money(value: float) -> str:
    if pd.isna(value):
        return "$0"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:.0f}"


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "0.0%"
    return f"{value * 100:.1f}%"


def tx_schema(df_tx: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "customer": infer_first_existing(df_tx.columns, ["customer_id", "customerid", "cust_id", "customer"]),
        "date": infer_first_existing(df_tx.columns, ["invoicedate", "invoice_date", "order_date", "date"]),
        "invoice": infer_first_existing(df_tx.columns, ["invoiceno", "invoice_no", "order_id", "invoice"]),
        "country": infer_first_existing(df_tx.columns, ["country"]),
        "product": infer_first_existing(df_tx.columns, ["description", "stockcode", "product", "item_name"]),
    }


def build_top_controls(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Render filters in a top toolbar (not sidebar) and return filtered df + control metadata.
    This duplicates the available filtering functionality but places controls in the page header.
    """
    c1, c2, c3, c4 = st.columns([3, 3, 3, 2])

    with c1:
        if st.button("Reload Data", key="top_reload"):
            st.cache_data.clear()
            st.experimental_rerun()
        uploaded = st.file_uploader("Optional: upload customer_segments.csv", type=["csv"], key="top_upload")

    with c2:
        segments = sorted(df["segment_name"].dropna().unique().tolist()) if "segment_name" in df.columns else []
        selected_segments = st.multiselect("Segments", options=segments, default=segments, key="top_segments")
        clusters = sorted(df["cluster"].dropna().astype(int).unique().tolist()) if "cluster" in df.columns else []
        selected_clusters = st.multiselect("Cluster IDs", options=clusters, default=clusters, key="top_clusters")

    with c3:
        churn_mode = st.radio("Churn Status", options=["All", "Churned only", "Active only"], index=0, key="top_churn_mode")
        selected_risk = st.multiselect(
            "Risk Bands",
            options=["Very Low", "Low", "Medium", "High", "Critical"],
            default=["Very Low", "Low", "Medium", "High", "Critical"],
            key="top_risk_bands",
        )

    with c4:
        top_n_watchlist = st.number_input("Watchlist Size", min_value=10, max_value=200, value=40, step=5, key="top_watchlist")
        risk_weight = st.slider("Priority Weight", min_value=0.1, max_value=0.9, value=0.7, step=0.05, key="top_weight")

    # Optional file upload handling replaces df if provided
    if uploaded is not None:
        external = pd.read_csv(uploaded)
        if "customer_id" in external.columns:
            external["customer_id"] = normalize_customer_id(external["customer_id"])
        for col in ["monetary", "frequency", "recency", "churn", "churn_probability", "cluster"]:
            if col in external.columns:
                external[col] = pd.to_numeric(external[col], errors="coerce")
        if "segment_name" not in external.columns and "cluster" in external.columns:
            external["segment_name"] = "Cluster " + external["cluster"].astype("Int64").astype(str)
        df = external

    working = add_risk_band(df)

    # Build filters (same logic as original build_filters)
    filtered = working.copy()
    if selected_segments and "segment_name" in filtered.columns:
        filtered = filtered[filtered["segment_name"].isin(selected_segments)]
    if selected_clusters and "cluster" in filtered.columns:
        filtered = filtered[filtered["cluster"].astype(int).isin(selected_clusters)]
    if selected_risk and "risk_band" in filtered.columns:
        filtered = filtered[filtered["risk_band"].isin(selected_risk)]

    min_prob = float(working["churn_probability"].min()) if "churn_probability" in working.columns else 0.0
    max_prob = float(working["churn_probability"].max()) if "churn_probability" in working.columns else 1.0
    prob_range = (max(0.0, min_prob), min(1.0, max_prob))
    if "churn_probability" in filtered.columns:
        filtered = filtered[(filtered["churn_probability"] >= prob_range[0]) & (filtered["churn_probability"] <= prob_range[1])]

    if "monetary" in filtered.columns:
        m_min = float(working["monetary"].min())
        m_max = float(working["monetary"].max())
        filtered = filtered[(filtered["monetary"] >= m_min) & (filtered["monetary"] <= m_max)]

    if "recency" in filtered.columns:
        r_min = int(working["recency"].min())
        r_max = int(working["recency"].max())
        filtered = filtered[(filtered["recency"] >= r_min) & (filtered["recency"] <= r_max)]

    if "frequency" in filtered.columns:
        f_min = int(working["frequency"].min())
        filtered = filtered[filtered["frequency"] >= f_min]

    if "churn" in filtered.columns:
        if churn_mode == "Churned only":
            filtered = filtered[filtered["churn"] == 1]
        elif churn_mode == "Active only":
            filtered = filtered[filtered["churn"] == 0]

    controls = {"top_n_watchlist": int(top_n_watchlist), "risk_weight": float(risk_weight)}
    return filtered, controls


def enrich_customers_with_tx(customer_df: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    schema = tx_schema(tx)
    c_col = schema["customer"]
    d_col = schema["date"]
    country_col = schema["country"]

    if c_col is None:
        return customer_df

    merged = customer_df.copy()
    merged["customer_key"] = normalize_customer_id(merged["customer_id"])

    tx_work = tx.copy()
    tx_work["customer_key"] = normalize_customer_id(tx_work[c_col])

    if d_col is not None:
        tx_work = tx_work[tx_work[d_col].notna()].copy()

    activity = (
        tx_work.groupby("customer_key")
        .agg(tx_rows=("customer_key", "size"))
        .reset_index()
    )

    if d_col is not None:
        dates = (
            tx_work.groupby("customer_key")
            .agg(first_order=(d_col, "min"), last_order=(d_col, "max"))
            .reset_index()
        )
        activity = activity.merge(dates, on="customer_key", how="left")

    if country_col is not None and "order_amount" in tx_work.columns:
        country = (
            tx_work.groupby(["customer_key", country_col], as_index=False)["order_amount"]
            .sum()
            .sort_values("order_amount", ascending=False)
            .drop_duplicates(subset=["customer_key"])
            .rename(columns={country_col: "dominant_country", "order_amount": "country_spend"})
        )
        activity = activity.merge(country[["customer_key", "dominant_country"]], on="customer_key", how="left")

    merged = merged.merge(activity, on="customer_key", how="left")
    return merged.drop(columns=["customer_key"])


def add_risk_band(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "churn_probability" not in out.columns:
        out["risk_band"] = "Unknown"
        return out

    out["risk_band"] = pd.cut(
        out["churn_probability"],
        bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["Very Low", "Low", "Medium", "High", "Critical"],
    ).astype(str)
    return out


def build_priority_table(df: pd.DataFrame, risk_weight: float = 0.7) -> pd.DataFrame:
    watch = df.copy()
    if watch.empty:
        watch["retention_priority"] = []
        return watch

    max_monetary = max(float(watch["monetary"].max()), 1.0)
    value_score = watch["monetary"] / max_monetary
    watch["retention_priority"] = (risk_weight * watch["churn_probability"]) + ((1.0 - risk_weight) * value_score)
    return watch.sort_values(["retention_priority", "churn_probability", "monetary"], ascending=[False, False, False])


def build_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    st.sidebar.title("⚙️ Filters & Controls")
    st.sidebar.caption("Configure your analysis parameters below.")

    if st.sidebar.button("Reload Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    uploaded = st.sidebar.file_uploader("Optional: upload customer_segments.csv", type=["csv"])

    if uploaded is not None:
        external = pd.read_csv(uploaded)
        if "customer_id" in external.columns:
            external["customer_id"] = normalize_customer_id(external["customer_id"])
        for col in ["monetary", "frequency", "recency", "churn", "churn_probability", "cluster"]:
            if col in external.columns:
                external[col] = pd.to_numeric(external[col], errors="coerce")
        if "segment_name" not in external.columns and "cluster" in external.columns:
            external["segment_name"] = "Cluster " + external["cluster"].astype("Int64").astype(str)
        df = external

    working = add_risk_band(df)

    segments = sorted(working["segment_name"].dropna().unique().tolist()) if "segment_name" in working.columns else []
    selected_segments = st.sidebar.multiselect("Segments", options=segments, default=segments)

    clusters = sorted(working["cluster"].dropna().astype(int).unique().tolist()) if "cluster" in working.columns else []
    selected_clusters = st.sidebar.multiselect("Cluster IDs", options=clusters, default=clusters)

    churn_mode = st.sidebar.radio("Churn Status", options=["All", "Churned only", "Active only"], index=0)

    selected_risk = st.sidebar.multiselect(
        "Risk Bands",
        options=["Very Low", "Low", "Medium", "High", "Critical"],
        default=["Very Low", "Low", "Medium", "High", "Critical"],
    )

    min_prob = float(working["churn_probability"].min()) if "churn_probability" in working.columns else 0.0
    max_prob = float(working["churn_probability"].max()) if "churn_probability" in working.columns else 1.0
    prob_range = st.sidebar.slider(
        "Churn Probability",
        min_value=0.0,
        max_value=1.0,
        value=(max(0.0, min_prob), min(1.0, max_prob)),
        step=0.01,
    )

    m_min = float(working["monetary"].min()) if "monetary" in working.columns else 0.0
    m_max = float(working["monetary"].max()) if "monetary" in working.columns else 1.0
    monetary_range = st.sidebar.slider(
        "Monetary Range",
        min_value=float(m_min),
        max_value=float(m_max),
        value=(float(m_min), float(m_max)),
        step=max(1.0, (m_max - m_min) / 250.0) if m_max > m_min else 1.0,
    )

    r_min = int(working["recency"].min()) if "recency" in working.columns else 0
    r_max = int(working["recency"].max()) if "recency" in working.columns else 1
    recency_range = st.sidebar.slider(
        "Recency (days)",
        min_value=r_min,
        max_value=r_max,
        value=(r_min, r_max),
        step=1,
    )

    f_min = int(working["frequency"].min()) if "frequency" in working.columns else 1
    f_max = int(working["frequency"].max()) if "frequency" in working.columns else 2
    min_frequency = st.sidebar.slider("Minimum Frequency", min_value=f_min, max_value=f_max, value=f_min, step=1)

    if "dominant_country" in working.columns:
        countries = sorted(working["dominant_country"].dropna().unique().tolist())
        selected_countries = st.sidebar.multiselect("Countries", options=countries, default=countries)
    else:
        selected_countries = []

    top_n_watchlist = st.sidebar.slider("Watchlist Size", min_value=10, max_value=200, value=40, step=5)
    risk_weight = st.sidebar.slider("Priority Weight (Risk vs Value)", min_value=0.1, max_value=0.9, value=0.7, step=0.05)

    filtered = working.copy()
    if selected_segments and "segment_name" in filtered.columns:
        filtered = filtered[filtered["segment_name"].isin(selected_segments)]
    if selected_clusters and "cluster" in filtered.columns:
        filtered = filtered[filtered["cluster"].astype(int).isin(selected_clusters)]
    if selected_risk and "risk_band" in filtered.columns:
        filtered = filtered[filtered["risk_band"].isin(selected_risk)]
    if "churn_probability" in filtered.columns:
        filtered = filtered[(filtered["churn_probability"] >= prob_range[0]) & (filtered["churn_probability"] <= prob_range[1])]
    if "monetary" in filtered.columns:
        filtered = filtered[(filtered["monetary"] >= monetary_range[0]) & (filtered["monetary"] <= monetary_range[1])]
    if "recency" in filtered.columns:
        filtered = filtered[(filtered["recency"] >= recency_range[0]) & (filtered["recency"] <= recency_range[1])]
    if "frequency" in filtered.columns:
        filtered = filtered[filtered["frequency"] >= min_frequency]

    if selected_countries and "dominant_country" in filtered.columns:
        filtered = filtered[filtered["dominant_country"].isin(selected_countries)]

    if "churn" in filtered.columns:
        if churn_mode == "Churned only":
            filtered = filtered[filtered["churn"] == 1]
        elif churn_mode == "Active only":
            filtered = filtered[filtered["churn"] == 0]

    controls = {
        "top_n_watchlist": top_n_watchlist,
        "risk_weight": risk_weight,
    }
    return filtered, controls


def show_kpis(df: pd.DataFrame) -> None:
    total_customers = int(df["customer_id"].nunique()) if "customer_id" in df.columns else len(df)
    total_revenue = float(df["monetary"].sum()) if "monetary" in df.columns else 0.0
    avg_prob = float(df["churn_probability"].mean()) if "churn_probability" in df.columns else 0.0
    churn_rate = float(df["churn"].mean()) if "churn" in df.columns else 0.0
    avg_freq = float(df["frequency"].mean()) if "frequency" in df.columns else 0.0
    critical = int((df["churn_probability"] >= 0.8).sum()) if "churn_probability" in df.columns else 0

    cards = [
        ("Customers", f"{total_customers:,}", "Population in current filter"),
        ("Revenue", fmt_money(total_revenue), "Total monetary value in scope"),
        ("Avg Churn Risk", fmt_pct(avg_prob), "Mean model risk score"),
        ("Observed Churn", fmt_pct(churn_rate), "Share of churn label = 1"),
        ("Avg Frequency", f"{avg_freq:.1f}", "Mean number of purchases"),
        ("Critical Risk", f"{critical:,}", "Customers with probability >= 0.80"),
    ]

    cols = st.columns(6)
    for col, (label, value, note) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def chart_segment_mix(df: pd.DataFrame) -> None:
    if "segment_name" not in df.columns:
        st.info("Segment data is unavailable.")
        return

    mix = (
        df.groupby("segment_name", dropna=False)
        .agg(customers=("customer_id", "nunique"), revenue=("monetary", "sum"), avg_risk=("churn_probability", "mean"))
        .sort_values("revenue", ascending=False)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(data=mix, x="revenue", y="segment_name", palette="YlGnBu", ax=ax)
    ax.set_title("Revenue by Segment")
    ax.set_xlabel("Revenue")
    ax.set_ylabel("Segment")
    ax.grid(alpha=0.15)
    st.pyplot(fig, use_container_width=True)


def chart_churn_distribution(df: pd.DataFrame) -> None:
    if "churn_probability" not in df.columns:
        st.info("Churn probability data is unavailable.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.histplot(
        data=df,
        x="churn_probability",
        hue="risk_band" if "risk_band" in df.columns else None,
        bins=28,
        alpha=0.32,
        multiple="stack",
        ax=ax,
    )
    ax.set_title("Risk Distribution")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Customers")
    ax.grid(alpha=0.12)
    st.pyplot(fig, use_container_width=True)


def chart_value_vs_recency(df: pd.DataFrame) -> None:
    needed_cols = {"recency", "monetary", "churn_probability"}
    if not needed_cols.issubset(df.columns):
        st.info("Recency, monetary, and churn_probability are required for this chart.")
        return

    plot_df = df.copy()
    if len(plot_df) > 2500:
        plot_df = plot_df.sample(2500, random_state=42)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=plot_df,
        x="recency",
        y="monetary",
        hue="segment_name" if "segment_name" in plot_df.columns else None,
        size="churn_probability",
        sizes=(16, 160),
        alpha=0.65,
        palette="viridis",
        ax=ax,
    )
    ax.set_title("Value vs Recency Map")
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Monetary")
    ax.grid(alpha=0.14)
    st.pyplot(fig, use_container_width=True)


def chart_monthly_trends(tx: pd.DataFrame) -> None:
    schema = tx_schema(tx)
    c_col = schema["customer"]
    d_col = schema["date"]

    if c_col is None or d_col is None or "order_amount" not in tx.columns:
        st.info("Missing transaction columns for trend analysis.")
        return

    monthly = tx[[c_col, d_col, "order_amount"]].dropna().copy()
    monthly["month"] = monthly[d_col].dt.to_period("M").dt.to_timestamp()

    trend = (
        monthly.groupby("month")
        .agg(revenue=("order_amount", "sum"), active_customers=(c_col, "nunique"))
        .reset_index()
        .sort_values("month")
    )

    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    sns.lineplot(data=trend, x="month", y="revenue", marker="o", color="#1b7a67", label="Revenue", ax=ax)
    ax2 = ax.twinx()
    sns.lineplot(data=trend, x="month", y="active_customers", marker="o", color="#cc8a1d", label="Active Customers", ax=ax2)

    ax.set_title("Monthly Revenue and Active Customer Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    ax2.set_ylabel("Active Customers")
    ax.grid(alpha=0.18)
    ax.tick_params(axis="x", rotation=32)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")
    st.pyplot(fig, use_container_width=True)


def chart_risk_heatmap(df: pd.DataFrame) -> None:
    needed = {"segment_name", "risk_band", "customer_id"}
    if not needed.issubset(df.columns):
        st.info("Segment and risk band columns are required for this chart.")
        return

    matrix = (
        df.groupby(["segment_name", "risk_band"])["customer_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    if matrix.empty:
        st.info("Not enough data for risk heatmap.")
        return

    matrix = matrix.reindex(columns=["Very Low", "Low", "Medium", "High", "Critical"], fill_value=0)
    share = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.heatmap(share, annot=True, fmt=".0%", cmap="YlOrBr", linewidths=0.4, cbar=True, ax=ax)
    ax.set_title("Risk Composition by Segment")
    ax.set_xlabel("Risk Band")
    ax.set_ylabel("Segment")
    st.pyplot(fig, use_container_width=True)


def build_cohort_retention(tx: pd.DataFrame) -> Optional[pd.DataFrame]:
    schema = tx_schema(tx)
    c_col = schema["customer"]
    d_col = schema["date"]
    if c_col is None or d_col is None:
        return None

    base = tx[[c_col, d_col]].dropna().copy()
    if base.empty:
        return None

    base["customer_key"] = normalize_customer_id(base[c_col])
    base["order_month"] = base[d_col].dt.to_period("M")
    base["cohort_month"] = base.groupby("customer_key")["order_month"].transform("min")
    base["cohort_index"] = (base["order_month"] - base["cohort_month"]).apply(lambda x: int(x.n))

    cohort_counts = (
        base.groupby(["cohort_month", "cohort_index"])["customer_key"]
        .nunique()
        .reset_index(name="customers")
    )

    pivot = cohort_counts.pivot(index="cohort_month", columns="cohort_index", values="customers").sort_index()
    if pivot.empty or 0 not in pivot.columns:
        return None

    retention = pivot.div(pivot[0], axis=0)
    retention.index = retention.index.astype(str)
    return retention.iloc[:, :13]


def chart_cohort_heatmap(tx: pd.DataFrame) -> None:
    retention = build_cohort_retention(tx)
    if retention is None:
        st.info("Cohort retention could not be built from available transaction columns.")
        return

    fig, ax = plt.subplots(figsize=(10, 5.2))
    sns.heatmap(retention, cmap="BuGn", annot=True, fmt=".0%", linewidths=0.35, cbar=True, ax=ax)
    ax.set_title("Monthly Cohort Retention")
    ax.set_xlabel("Months Since First Purchase")
    ax.set_ylabel("Cohort Month")
    st.pyplot(fig, use_container_width=True)


def chart_top_products(tx: pd.DataFrame) -> None:
    schema = tx_schema(tx)
    p_col = schema["product"]
    if p_col is None or "order_amount" not in tx.columns:
        st.info("Product and order_amount columns are required for product analysis.")
        return

    top = (
        tx[[p_col, "order_amount"]]
        .dropna()
        .groupby(p_col, as_index=False)["order_amount"]
        .sum()
        .sort_values("order_amount", ascending=False)
        .head(12)
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    sns.barplot(data=top, x="order_amount", y=p_col, palette="mako", ax=ax)
    ax.set_title("Top Products by Revenue")
    ax.set_xlabel("Revenue")
    ax.set_ylabel("Product")
    ax.grid(alpha=0.15)
    st.pyplot(fig, use_container_width=True)


def chart_country_mix(df: pd.DataFrame) -> None:
    if "dominant_country" not in df.columns:
        st.info("Country information is unavailable.")
        return

    summary = (
        df.dropna(subset=["dominant_country"])
        .groupby("dominant_country", as_index=False)
        .agg(customers=("customer_id", "nunique"), revenue=("monetary", "sum"))
        .sort_values("revenue", ascending=False)
        .head(10)
    )
    if summary.empty:
        st.info("No country-level records available after filtering.")
        return

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    sns.barplot(data=summary, x="revenue", y="dominant_country", palette="crest", ax=ax)
    ax.set_title("Top Countries by Revenue")
    ax.set_xlabel("Revenue")
    ax.set_ylabel("Country")
    ax.grid(alpha=0.14)
    st.pyplot(fig, use_container_width=True)


def show_insights(df: pd.DataFrame) -> None:
    needed = {"segment_name", "churn_probability", "monetary", "customer_id"}
    if not needed.issubset(df.columns):
        return

    summary = (
        df.groupby("segment_name", as_index=False)
        .agg(customers=("customer_id", "nunique"), avg_risk=("churn_probability", "mean"), avg_value=("monetary", "mean"))
    )
    if summary.empty:
        return

    high_risk = summary.sort_values("avg_risk", ascending=False).iloc[0]
    high_value = summary.sort_values("avg_value", ascending=False).iloc[0]
    large = summary.sort_values("customers", ascending=False).iloc[0]

    st.markdown(
        f"""
        <div class="insight-card">
            <strong>Highest risk segment:</strong> {high_risk['segment_name']}<br>
            Avg churn risk: {fmt_pct(float(high_risk['avg_risk']))} across {int(high_risk['customers'])} customers.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="insight-card">
            <strong>Highest value segment:</strong> {high_value['segment_name']}<br>
            Avg monetary value: {fmt_money(float(high_value['avg_value']))}.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="insight-card">
            <strong>Largest segment by count:</strong> {large['segment_name']}<br>
            Total customers: {int(large['customers'])}. Prioritize scalable retention automation here.
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_watchlist(df: pd.DataFrame, top_n: int, risk_weight: float) -> pd.DataFrame:
    needed = {"customer_id", "segment_name", "churn_probability", "monetary", "recency", "frequency"}
    if not needed.issubset(df.columns):
        st.info("Watchlist requires customer, segment, churn_probability, monetary, recency, and frequency columns.")
        return pd.DataFrame()

    watch = build_priority_table(df, risk_weight=risk_weight).head(top_n).copy()

    cols = [
        "customer_id",
        "segment_name",
        "risk_band",
        "churn_probability",
        "monetary",
        "recency",
        "frequency",
        "retention_priority",
    ]
    cols = [col for col in cols if col in watch.columns]

    st.dataframe(
        watch[cols],
        use_container_width=True,
        hide_index=True,
        height=430,
        column_config={
            "churn_probability": st.column_config.ProgressColumn(
                "Churn Probability", format="%.2f", min_value=0, max_value=1
            ),
            "retention_priority": st.column_config.ProgressColumn(
                "Retention Priority", format="%.2f", min_value=0, max_value=1
            ),
            "monetary": st.column_config.NumberColumn("Monetary", format="$ %.2f"),
        },
    )

    st.download_button(
        "Download Retention Watchlist",
        data=watch[cols].to_csv(index=False).encode("utf-8"),
        file_name="retention_watchlist.csv",
        mime="text/csv",
        use_container_width=True,
    )
    return watch


def show_simulator(df: pd.DataFrame, risk_weight: float) -> None:
    needed = {"customer_id", "churn_probability", "monetary"}
    if not needed.issubset(df.columns):
        st.info("Simulator requires customer_id, churn_probability, and monetary columns.")
        return

    ranked = build_priority_table(df, risk_weight=risk_weight)
    if ranked.empty:
        st.info("No customers available for simulation in current filters.")
        return

    max_target = min(len(ranked), 1200)
    if max_target <= 10:
        target_size = int(
            st.number_input(
                "Target customers",
                min_value=1,
                max_value=max_target,
                value=max_target,
                step=1,
            )
        )
        st.caption("Small filtered cohort detected: targeting supports up to available customers.")
    else:
        target_size = st.slider(
            "Target customers",
            min_value=10,
            max_value=max_target,
            value=min(200, max_target),
            step=10,
        )
    risk_reduction = st.slider("Expected risk reduction per targeted customer", min_value=0.01, max_value=0.80, value=0.15, step=0.01)
    gross_margin = st.slider("Gross margin on saved revenue", min_value=0.10, max_value=0.90, value=0.35, step=0.05)

    c1, c2 = st.columns(2)
    with c1:
        contact_cost = st.number_input("Contact cost per customer", min_value=0.0, value=1.5, step=0.5)
    with c2:
        incentive_cost = st.number_input("Offer cost per customer", min_value=0.0, value=6.0, step=0.5)

    target = ranked.head(target_size).copy()
    expected_drop = np.minimum(target["churn_probability"].values, risk_reduction)

    saved_customers = float(expected_drop.sum())
    saved_revenue = float((target["monetary"].values * expected_drop).sum())
    saved_profit = saved_revenue * gross_margin
    campaign_cost = target_size * (contact_cost + incentive_cost)
    net_profit = saved_profit - campaign_cost
    roi = (net_profit / campaign_cost) if campaign_cost > 0 else np.nan

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Expected Customers Saved", f"{saved_customers:.1f}")
    m2.metric("Recovered Revenue", fmt_money(saved_revenue))
    m3.metric("Campaign Cost", fmt_money(campaign_cost))
    m4.metric("ROI", fmt_pct(roi) if np.isfinite(roi) else "n/a")

    bars = pd.DataFrame(
        {
            "metric": ["Saved Gross Profit", "Campaign Cost", "Net Profit"],
            "value": [saved_profit, campaign_cost, net_profit],
        }
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sns.barplot(data=bars, x="metric", y="value", palette=["#1b7a67", "#cc8a1d", "#2c3e50"], ax=ax)
    ax.set_title("Scenario Financial Summary")
    ax.set_xlabel("")
    ax.set_ylabel("Amount")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(alpha=0.15)
    st.pyplot(fig, use_container_width=True)

    st.caption("Assumption: revenue recovered is proportional to probability reduction among targeted customers.")


def main() -> None:
    inject_styles()

    st.markdown(
        """
        <div class="hero">
            <div class="subtitle">🎯 RetentionPro</div>
            <h1>Advanced Analytics Hub</h1>
            <p>
                Next-generation customer intelligence platform powered by AI. Segment customers, predict churn, 
                simulate campaigns, and execute data-driven retention strategies at scale.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data = load_data()
    if "customer_segments" not in data:
        st.error("customer_segments.csv was not found. Run customer_segmentation_retention.py first.")
        return

    customer_df = data["customer_segments"].copy()
    if "clean_transactions" in data:
        customer_df = enrich_customers_with_tx(customer_df, data["clean_transactions"])

    # Use the new top toolbar controls (completely different layout)
    filtered_df, controls = build_top_controls(customer_df)
    if filtered_df.empty:
        st.warning("No rows match current filters. Widen ranges or include more segments.")
        return

    # NEW: compact top navbar (horizontal radio) controls page navigation
    pages = [
        "📊 Dashboard",
        "🔍 Segmentation",
        "📈 Trends & Products",
        "🎯 Campaign Simulator",
        "📋 Action Plan",
        "🗂️ Data Explorer",
    ]
    selected_page = st.radio("", options=pages, index=0, horizontal=True, key="page_nav")

    # Show KPIs always under the navbar for a consistent layout
    show_kpis(filtered_df)

    # Layout: main content area with a right-hand compact control panel (not the Streamlit sidebar)
    main_col, control_col = st.columns([3, 1])

    with main_col:
        if selected_page == pages[0]:
            # Dashboard
            c_left, c_right = st.columns([1, 1])
            with c_left:
                chart_segment_mix(filtered_df)
            with c_right:
                chart_churn_distribution(filtered_df)

            if "clean_transactions" in data:
                chart_monthly_trends(data["clean_transactions"])
            else:
                st.info("clean_transactions.csv is missing, so monthly trends are unavailable.")

        elif selected_page == pages[1]:
            st.markdown('<div class="section-header">Customer Segmentation Analysis</div>', unsafe_allow_html=True)
            lft, rgt = st.columns([1.05, 0.95])
            with lft:
                chart_value_vs_recency(filtered_df)
            with rgt:
                show_insights(filtered_df)
                chart_country_mix(filtered_df)

            if "segment_summary" in data:
                st.markdown('<div class="section-header">Segment Summary</div>', unsafe_allow_html=True)
                st.dataframe(data["segment_summary"], use_container_width=True, hide_index=True)

            chart_risk_heatmap(filtered_df)

        elif selected_page == pages[2]:
            st.markdown('<div class="section-header">Cohort Retention & Product Performance</div>', unsafe_allow_html=True)
            if "clean_transactions" in data:
                c1, c2 = st.columns([1.05, 0.95])
                with c1:
                    chart_cohort_heatmap(data["clean_transactions"])
                with c2:
                    chart_top_products(data["clean_transactions"])
            else:
                st.info("clean_transactions.csv is required for cohort and product analysis.")

        elif selected_page == pages[3]:
            st.markdown('<div class="section-header">Retention Campaign Simulator</div>', unsafe_allow_html=True)
            st.caption("Test retention campaign assumptions and calculate ROI before execution.")
            show_simulator(filtered_df, risk_weight=float(controls["risk_weight"]))

        elif selected_page == pages[4]:
            st.markdown('<div class="section-header">Retention Action Plan</div>', unsafe_allow_html=True)
            st.caption("Prioritized watchlist of high-risk customers by risk and value score.")
            watch = show_watchlist(
                filtered_df,
                top_n=int(controls["top_n_watchlist"]),
                risk_weight=float(controls["risk_weight"]),
            )

            if not watch.empty and "segment_name" in watch.columns:
                st.markdown('<div class="section-header">Segment-Level Action Plan</div>', unsafe_allow_html=True)
                plan = (
                    watch.groupby("segment_name", as_index=False)
                    .agg(
                        target_customers=("customer_id", "nunique"),
                        avg_risk=("churn_probability", "mean"),
                        target_value=("monetary", "sum"),
                    )
                    .sort_values(["avg_risk", "target_value"], ascending=[False, False])
                )
                st.dataframe(plan, use_container_width=True, hide_index=True)

        elif selected_page == pages[5]:
            st.markdown('<div class="section-header">Filtered Customer Data</div>', unsafe_allow_html=True)
            st.caption("Export and explore the full customer dataset with current filters applied.")
            st.dataframe(filtered_df, use_container_width=True, hide_index=True, height=520)
            st.download_button(
                "Download Filtered Customers",
                data=filtered_df.to_csv(index=False).encode("utf-8"),
                file_name="filtered_customers.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with control_col:
        # Compact control card on the right of the main content (different from previous sidebar)
        st.markdown('<div style="background-color: var(--dark-tertiary); padding: 12px; border-radius: 10px; border:1px solid var(--border-color);">', unsafe_allow_html=True)
        st.markdown('<strong style="color:var(--accent-cyan);">Filters & Controls</strong>', unsafe_allow_html=True)
        st.caption('Compact control panel')
        # Mirror a few high-value controls here for quick access
        segs = sorted(filtered_df["segment_name"].dropna().unique().tolist()) if "segment_name" in filtered_df.columns else []
        st.multiselect("Segments", options=segs, default=segs, key="ctrl_segments")
        st.slider("Priority Weight", min_value=0.1, max_value=0.9, value=float(controls["risk_weight"]), step=0.05, key="ctrl_weight")
        st.number_input("Watchlist size", min_value=10, max_value=200, value=int(controls["top_n_watchlist"]), key="ctrl_watch")
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
