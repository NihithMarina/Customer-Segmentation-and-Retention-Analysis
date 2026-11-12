import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")


# -----------------------------
# Configuration
# -----------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Prefer customers.csv (matches prompt) if it exists, else fall back to online_retail.csv
DATA_PATH = (
    os.path.join(PROJECT_DIR, "customers.csv")
    if os.path.exists(os.path.join(PROJECT_DIR, "customers.csv"))
    else os.path.join(PROJECT_DIR, "online_retail.csv")
)

RANDOM_STATE = 42

# Outputs: saved in the same folder (no folder structure changes)
OUT_CLEAN_TXN = os.path.join(PROJECT_DIR, "clean_transactions.csv")
OUT_CUSTOMER_SEGMENTS = os.path.join(PROJECT_DIR, "customer_segments.csv")
OUT_SEGMENT_SUMMARY = os.path.join(PROJECT_DIR, "segment_summary.csv")
OUT_CHURN_PREDICTIONS = os.path.join(PROJECT_DIR, "churn_predictions.csv")

PLOT_ELBOW = os.path.join(PROJECT_DIR, "plot_elbow.png")
PLOT_PCA = os.path.join(PROJECT_DIR, "plot_clusters_pca.png")
PLOT_CHURN_ROC = os.path.join(PROJECT_DIR, "plot_churn_roc.png")
PLOT_SEGMENT_CHURN = os.path.join(PROJECT_DIR, "plot_segment_churn.png")


# -----------------------------
# Step 1: Data loading & cleaning
# -----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def infer_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Numeric columns: fill with median (robust to outliers)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # Non-numeric columns: fill with mode (most common)
    non_num_cols = [c for c in df.columns if c not in num_cols]
    for c in non_num_cols:
        if df[c].isna().any():
            mode = df[c].mode(dropna=True)
            fill_value = mode.iloc[0] if len(mode) > 0 else "unknown"
            df[c] = df[c].fillna(fill_value)

    return df


def load_and_clean_transactions(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Loads a CSV and returns a clean transaction-level dataset.

    Supports:
    - Prompt-style customer datasets (customers.csv)
    - Transactional retail datasets (like Online Retail)

    Output invariants:
    - Must contain: customer_id, order_date, order_amount
    """
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    # Infer columns from either customer-style or online-retail-style data
    col_customer = infer_column(df, ["customer_id", "customerid", "cust_id", "customer"])
    col_order_date = infer_column(df, ["order_date", "invoice_date", "invoicedate", "purchase_date", "date"])
    col_invoice = infer_column(df, ["order_id", "invoice_no", "invoiceno", "invoice", "transaction_id"])

    col_quantity = infer_column(df, ["quantity", "qty"])
    col_unit_price = infer_column(df, ["unitprice", "unit_price", "price"])

    # Minimum requirements: customer + a date
    if col_customer is None:
        raise ValueError(
            "Could not infer customer identifier column. Expected e.g. customer_id/CustomerID. "
            f"Found: {list(df.columns)}"
        )
    if col_order_date is None:
        raise ValueError(
            "Could not infer order date column. Expected e.g. order_date/InvoiceDate. "
            f"Found: {list(df.columns)}"
        )

    # Parse date
    df[col_order_date] = pd.to_datetime(df[col_order_date], errors="coerce")

    # Standardize customer_id (Online Retail often stores it as float)
    df[col_customer] = (
        df[col_customer]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )

    # Drop rows missing core identifiers (can't impute these safely)
    df = df[df[col_customer].notna() & (df[col_customer] != "")].copy()
    df = df[df[col_order_date].notna()].copy()

    # Coerce numeric columns if present
    if "order_amount" in df.columns:
        df["order_amount"] = pd.to_numeric(df["order_amount"], errors="coerce")

    if col_quantity is not None:
        df[col_quantity] = pd.to_numeric(df[col_quantity], errors="coerce")
    if col_unit_price is not None:
        df[col_unit_price] = pd.to_numeric(df[col_unit_price], errors="coerce")

    # Fill missing values (safe for numeric/categorical)
    df = fill_missing_values(df)

    # Drop exact duplicates
    df = df.drop_duplicates().copy()

    # Remove common return/cancellation patterns in online retail datasets
    if col_invoice is not None and col_invoice in df.columns:
        df[col_invoice] = df[col_invoice].astype(str).str.strip()
        df = df[~df[col_invoice].str.startswith("C", na=False)].copy()

    # Create order_amount if needed
    if "order_amount" not in df.columns:
        if col_quantity is None or col_unit_price is None:
            raise ValueError(
                "Cannot compute order_amount. Expected either existing order_amount column "
                "or Quantity + UnitPrice columns."
            )
        df["order_amount"] = df[col_quantity] * df[col_unit_price]

    # Remove non-sensical values
    df = df[df["order_amount"] >= 0].copy()

    # Keep a consistent minimal schema plus any extra columns for possible profiling
    df = df.copy()

    columns = {
        "customer_id": col_customer,
        "order_date": col_order_date,
        "invoice_id": col_invoice if col_invoice is not None else "",
    }

    return df, columns


# -----------------------------
# Step 2: Feature engineering (RFM)
# -----------------------------
def build_rfm(df_txn: pd.DataFrame, columns: Dict[str, str]) -> pd.DataFrame:
    """Build customer-level RFM table.

    RFM definitions (interview-friendly):
    - Recency: days since last purchase (lower is better)
    - Frequency: number of purchases/orders (higher is better)
    - Monetary: total spend (higher is better)
    """
    customer_col = columns["customer_id"]
    date_col = columns["order_date"]
    invoice_col = columns.get("invoice_id", "")

    # Reference date: one day after the last observed order
    reference_date = df_txn[date_col].max() + pd.Timedelta(days=1)

    # Frequency: count unique invoices when available (more stable than counting line items)
    if invoice_col and invoice_col in df_txn.columns:
        frequency = df_txn.groupby(customer_col)[invoice_col].nunique()
    else:
        frequency = df_txn.groupby(customer_col).size()

    agg = df_txn.groupby(customer_col).agg(
        last_purchase_date=(date_col, "max"),
        first_purchase_date=(date_col, "min"),
        monetary=("order_amount", "sum"),
    )

    rfm = agg.copy()
    rfm["frequency"] = frequency

    # Recency is computed from reference_date so it is relative and easy to interpret
    rfm["recency"] = (reference_date - rfm["last_purchase_date"]).dt.days

    # Tenure gives customer lifetime within the data window
    rfm["tenure_days"] = (reference_date - rfm["first_purchase_date"]).dt.days

    # Helpful extra feature: average order value
    rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"].replace(0, np.nan)
    rfm["avg_order_value"] = rfm["avg_order_value"].fillna(0.0)

    # Clean up types and names
    rfm = rfm.reset_index().rename(columns={customer_col: "customer_id"})
    rfm["customer_id"] = rfm["customer_id"].astype(str)

    # Sanity clamps
    rfm["recency"] = rfm["recency"].clip(lower=0)
    rfm["frequency"] = rfm["frequency"].clip(lower=1)
    rfm["monetary"] = rfm["monetary"].clip(lower=0)

    return rfm


def scale_features(rfm: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[feature_cols])
    return X, scaler


# -----------------------------
# Step 3: Customer segmentation (KMeans)
# -----------------------------
def compute_elbow_inertia(X: np.ndarray, k_min: int = 2, k_max: int = 10) -> Tuple[List[int], List[float]]:
    ks = list(range(k_min, k_max + 1))
    inertias: List[float] = []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X)
        inertias.append(float(km.inertia_))

    return ks, inertias


def choose_k_by_simple_elbow(ks: List[int], inertias: List[float]) -> int:
    """A deterministic elbow heuristic using second differences.

    This avoids manual selection while still being explainable.
    """
    if len(ks) < 3:
        return int(ks[0])

    y = np.array(inertias, dtype=float)
    second_diff = np.diff(y, n=2)

    # Strongest bend = max absolute second difference
    elbow_idx = int(np.argmax(np.abs(second_diff))) + 1
    k = int(ks[elbow_idx])

    # Keep k in a reasonable range for interview storytelling
    return int(np.clip(k, 2, 8))


def plot_elbow(ks: List[int], inertias: List[float], chosen_k: int, path: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, marker="o")
    plt.axvline(chosen_k, color="red", linestyle="--", label=f"Chosen k={chosen_k}")
    plt.title("Elbow Method (KMeans)")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def fit_kmeans(X: np.ndarray, k: int) -> KMeans:
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    model.fit(X)
    return model


def plot_clusters_pca(X: np.ndarray, labels: np.ndarray, path: str) -> None:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X2 = pca.fit_transform(X)

    plot_df = pd.DataFrame({
        "pc1": X2[:, 0],
        "pc2": X2[:, 1],
        "cluster": labels.astype(int),
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="pc1", y="pc2", hue="cluster", palette="tab10", s=40)
    plt.title("Customer Clusters (PCA 2D)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# -----------------------------
# Step 4: Segment profiling
# -----------------------------
def profile_segments(customer_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        customer_df
        .groupby("cluster")
        .agg(
            customers=("customer_id", "nunique"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .reset_index()
    )

    return summary


def assign_segment_names(summary_df: pd.DataFrame) -> Dict[int, str]:
    """Map each cluster to a human-readable segment name.

    Simple, explainable rules based on relative ranking:
    - High monetary + high frequency + low recency -> High-Value Loyal
    - High recency (stale) -> At-Risk
    - Low frequency + low monetary -> Low-Engagement
    - Else -> Regular
    """
    df = summary_df.copy()

    # Normalize rankings (percentiles across clusters)
    df["p_recency"] = df["avg_recency"].rank(pct=True)
    df["p_frequency"] = df["avg_frequency"].rank(pct=True)
    df["p_monetary"] = df["avg_monetary"].rank(pct=True)

    mapping: Dict[int, str] = {}
    for _, row in df.iterrows():
        cluster = int(row["cluster"])

        high_value = (row["p_monetary"] >= 0.75) and (row["p_frequency"] >= 0.75)
        loyal = row["p_recency"] <= 0.25
        at_risk = row["p_recency"] >= 0.75
        low_engagement = (row["p_frequency"] <= 0.25) and (row["p_monetary"] <= 0.25)

        if high_value and loyal:
            name = "High-Value Loyal Customers"
        elif at_risk:
            name = "At-Risk Customers"
        elif low_engagement:
            name = "Low-Engagement Customers"
        else:
            name = "Regular Customers"

        mapping[cluster] = name

    return mapping


# -----------------------------
# Step 5: Retention / churn analysis
# -----------------------------
def create_or_use_churn(customer_df: pd.DataFrame, churn_col: Optional[str] = None) -> pd.DataFrame:
    df = customer_df.copy()

    if churn_col is not None and churn_col in df.columns:
        df["churn"] = pd.to_numeric(df[churn_col], errors="coerce").fillna(0).astype(int)
        return df

    # If churn doesn't exist, define it logically:
    # Customers with unusually high recency (stale) are treated as churned.
    threshold = float(df["recency"].quantile(0.75))
    df["churn"] = (df["recency"] > threshold).astype(int)
    df["churn_definition"] = f"recency > {threshold:.0f} days (75th percentile)"
    return df


def train_churn_model(customer_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float], object, List[str]]:
    """Train a simple churn model and return churn probabilities.

    Uses a train/test split. Models:
    - LogisticRegression: interpretable baseline
    - RandomForest: fallback if LogisticRegression fails (rare)
    """
    df = customer_df.copy()

    feature_cols = [
        "recency",
        "frequency",
        "monetary",
        "tenure_days",
        "avg_order_value",
        "cluster",
    ]

    X = df[feature_cols].copy()
    y = df["churn"].astype(int)

    # Ensure numeric
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    try:
        model.fit(X_train, y_train)
        proba_test = model.predict_proba(X_test)[:, 1]
    except Exception:
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)
        proba_test = model.predict_proba(X_test)[:, 1]

    pred_test = (proba_test >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred_test)),
        "precision": float(precision_score(y_test, pred_test, zero_division=0)),
        "recall": float(recall_score(y_test, pred_test, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba_test)) if len(np.unique(y_test)) > 1 else float("nan"),
    }

    # Fit on all data for final probabilities
    model.fit(X, y)
    df["churn_probability"] = model.predict_proba(X)[:, 1]

    return df, metrics, model, feature_cols


def plot_roc_curve(customer_df: pd.DataFrame, path: str) -> None:
    y_true = customer_df["churn"].astype(int)
    y_score = customer_df["churn_probability"].astype(float)

    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("Churn Model ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_segment_churn(customer_df: pd.DataFrame, path: str) -> None:
    plot_df = (
        customer_df
        .groupby(["segment_name"])
        .agg(churn_rate=("churn", "mean"), customers=("customer_id", "nunique"))
        .reset_index()
        .sort_values("churn_rate", ascending=False)
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="churn_rate", y="segment_name", color="#4C72B0")
    plt.title("Churn Rate by Segment")
    plt.xlabel("Churn rate")
    plt.ylabel("Segment")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# -----------------------------
# Step 6: Business insights
# -----------------------------
def print_business_insights(customer_df: pd.DataFrame, segment_summary: pd.DataFrame, metrics: Dict[str, float], model, feature_cols: List[str]) -> None:
    print("\n" + "=" * 70)
    print("BUSINESS INSIGHTS (plain English)")
    print("=" * 70)

    # Segment churn comparison
    seg_churn = (
        customer_df
        .groupby("segment_name")
        .agg(churn_rate=("churn", "mean"), customers=("customer_id", "nunique"))
        .reset_index()
        .sort_values("churn_rate", ascending=False)
    )

    worst = seg_churn.iloc[0]
    best = seg_churn.iloc[-1]

    print(f"1) Highest churn segment: {worst['segment_name']} (churn_rate={worst['churn_rate']:.2%}, customers={int(worst['customers'])})")
    print(f"2) Lowest churn segment:  {best['segment_name']} (churn_rate={best['churn_rate']:.2%}, customers={int(best['customers'])})")

    # Top customers to target: high churn probability + high monetary
    target = (
        customer_df
        .sort_values(["churn_probability", "monetary"], ascending=[False, False])
        .head(10)
        [["customer_id", "segment_name", "churn_probability", "recency", "frequency", "monetary"]]
    )

    print("\n3) Top 10 customers to target for retention (high churn risk + high spend):")
    for _, r in target.iterrows():
        print(
            f"   - customer_id={r['customer_id']}, segment={r['segment_name']}, "
            f"churn_prob={r['churn_probability']:.2f}, recency={int(r['recency'])}, "
            f"freq={int(r['frequency'])}, spend={r['monetary']:.2f}"
        )

    # Model metrics
    print("\n4) Churn model performance (hold-out test):")
    for k, v in metrics.items():
        print(f"   - {k}: {v:.3f}" if np.isfinite(v) else f"   - {k}: n/a")

    # Feature influence (basic, interview-ready)
    print("\n5) What drives churn (model-based signal):")

    if hasattr(model, "coef_"):
        coefs = model.coef_.ravel()
        influence = pd.DataFrame({"feature": feature_cols, "coef": coefs}).sort_values("coef", ascending=False)
        top_pos = influence.head(3)
        top_neg = influence.tail(3)

        print("   Features increasing churn risk (positive coefficients):")
        for _, r in top_pos.iterrows():
            print(f"   - {r['feature']}: coef={r['coef']:.3f}")

        print("   Features decreasing churn risk (negative coefficients):")
        for _, r in top_neg.iterrows():
            print(f"   - {r['feature']}: coef={r['coef']:.3f}")

        print("   Interpretation tip: higher recency usually increases churn risk; higher frequency often reduces it.")

    elif hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        influence = pd.DataFrame({"feature": feature_cols, "importance": imps}).sort_values("importance", ascending=False)
        print("   Most important features (RandomForest):")
        for _, r in influence.head(5).iterrows():
            print(f"   - {r['feature']}: importance={r['importance']:.3f}")

    print("\n6) Recommended actions:")
    print("   - For At-Risk customers: send win-back offers, personalized reminders, and replenishment prompts.")
    print("   - For Low-Engagement customers: reduce friction (bundles, onboarding emails) and test price incentives.")
    print("   - For High-Value Loyal customers: protect with VIP benefits and early access (prevent silent churn).")


# -----------------------------
# Step 7: End-to-end run
# -----------------------------
def main() -> None:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found. Looked for: {DATA_PATH}")

    print(f"Using dataset: {DATA_PATH}")

    # 1) Load and clean
    df_txn, cols = load_and_clean_transactions(DATA_PATH)
    df_txn.to_csv(OUT_CLEAN_TXN, index=False)

    # 2) RFM features
    rfm = build_rfm(df_txn, cols)

    # 3) KMeans segmentation
    rfm_features = ["recency", "frequency", "monetary"]
    X_scaled, _ = scale_features(rfm, rfm_features)

    ks, inertias = compute_elbow_inertia(X_scaled, k_min=2, k_max=10)
    chosen_k = choose_k_by_simple_elbow(ks, inertias)
    plot_elbow(ks, inertias, chosen_k, PLOT_ELBOW)

    kmeans = fit_kmeans(X_scaled, chosen_k)
    rfm["cluster"] = kmeans.labels_.astype(int)

    plot_clusters_pca(X_scaled, rfm["cluster"].values, PLOT_PCA)

    # 4) Segment profiling + naming
    segment_summary = profile_segments(rfm)
    name_map = assign_segment_names(segment_summary)
    rfm["segment_name"] = rfm["cluster"].map(name_map)

    # Store segment summary with names (helpful for interview explanation)
    segment_summary["segment_name"] = segment_summary["cluster"].map(name_map)
    segment_summary = segment_summary[[
        "cluster",
        "segment_name",
        "customers",
        "avg_recency",
        "avg_frequency",
        "avg_monetary",
    ]].sort_values("avg_monetary", ascending=False)

    segment_summary.to_csv(OUT_SEGMENT_SUMMARY, index=False)

    # 5) Churn definition + churn model
    churn_col = infer_column(rfm, ["churn", "is_churn", "churned"])  # usually absent here
    customer_df = create_or_use_churn(rfm, churn_col=churn_col)

    customer_df, metrics, model, feature_cols = train_churn_model(customer_df)

    plot_roc_curve(customer_df, PLOT_CHURN_ROC)
    plot_segment_churn(customer_df, PLOT_SEGMENT_CHURN)

    # 6) Insights
    print_business_insights(customer_df, segment_summary, metrics, model, feature_cols)

    # 7) Outputs
    customer_df.to_csv(OUT_CUSTOMER_SEGMENTS, index=False)
    customer_df[["customer_id", "churn", "churn_probability"]].to_csv(OUT_CHURN_PREDICTIONS, index=False)

    print("\n" + "=" * 70)
    print("SAVED OUTPUTS")
    print("=" * 70)
    print(f"- Clean transactions:     {OUT_CLEAN_TXN}")
    print(f"- Customer segments:      {OUT_CUSTOMER_SEGMENTS}")
    print(f"- Segment summary:        {OUT_SEGMENT_SUMMARY}")
    print(f"- Churn predictions:      {OUT_CHURN_PREDICTIONS}")
    print(f"- Plot (elbow):           {PLOT_ELBOW}")
    print(f"- Plot (clusters PCA):    {PLOT_PCA}")
    print(f"- Plot (churn ROC):       {PLOT_CHURN_ROC}")
    print(f"- Plot (segment churn):   {PLOT_SEGMENT_CHURN}")


if __name__ == "__main__":
    main()
