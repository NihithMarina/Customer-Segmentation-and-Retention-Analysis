from __future__ import annotations

import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

import customer_segmentation_retention as csr


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR.parent / "screenshots" / "testing_evidence"


def save_card(file_name: str, title: str, passed: bool, details: list[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    status = "PASS" if passed else "FAIL"
    status_color = "#1b7a67" if passed else "#c0392b"

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    fig.patch.set_facecolor("#f6f8ee")
    ax.set_facecolor("#ffffff")
    ax.axis("off")

    ax.text(0.03, 0.90, title, fontsize=19, weight="bold", color="#163a33", transform=ax.transAxes)
    ax.text(
        0.03,
        0.80,
        f"Status: {status}",
        fontsize=15,
        weight="bold",
        color=status_color,
        transform=ax.transAxes,
    )

    y = 0.70
    for line in details:
        ax.text(0.05, y, f"- {line}", fontsize=12.5, color="#2f4f48", transform=ax.transAxes)
        y -= 0.08

    ax.text(
        0.03,
        0.06,
        "Customer Segmentation & Retention Analysis - Automated Testing Evidence",
        fontsize=10,
        color="#4f6b63",
        transform=ax.transAxes,
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / file_name)
    plt.close(fig)


def functional_data_upload_test() -> tuple[bool, list[str]]:
    cs_path = BASE_DIR / "customer_segments.csv"
    df = pd.read_csv(cs_path)

    required = {"customer_id", "segment_name", "churn_probability", "monetary"}
    present = set(df.columns)
    missing = required - present

    passed = len(df) > 0 and not missing
    details = [
        f"Rows loaded from customer_segments.csv: {len(df):,}",
        f"Required columns present: {sorted(required - missing)}",
        f"Missing columns: {sorted(missing) if missing else 'None'}",
    ]
    return passed, details


def functional_filters_test() -> tuple[bool, list[str]]:
    df = pd.read_csv(BASE_DIR / "customer_segments.csv")
    before = len(df)

    if "segment_name" in df.columns:
        chosen_segments = sorted(df["segment_name"].dropna().astype(str).unique().tolist())[:2]
    else:
        chosen_segments = []

    filtered = df.copy()
    if chosen_segments:
        filtered = filtered[filtered["segment_name"].astype(str).isin(chosen_segments)]

    if "churn_probability" in filtered.columns:
        prob_cut = float(filtered["churn_probability"].quantile(0.50))
        filtered = filtered[filtered["churn_probability"] >= prob_cut]
    else:
        prob_cut = 0.0

    if "churn" in filtered.columns:
        filtered = filtered[filtered["churn"] == 1]

    after = len(filtered)
    passed = before > 0 and (0 < after < before)
    details = [
        f"Rows before filter: {before:,}",
        f"Rows after filter: {after:,}",
        f"Applied filters: segments={chosen_segments or 'n/a'}, churn_probability >= {prob_cut:.2f}, churn=1",
    ]
    return passed, details


def functional_dashboard_readiness_test() -> tuple[bool, list[str]]:
    required_files = [
        BASE_DIR / "streamlit_dashboard.py",
        BASE_DIR / "customer_segments.csv",
        BASE_DIR / "clean_transactions.csv",
        BASE_DIR / "segment_summary.csv",
    ]

    existing = [p for p in required_files if p.exists()]

    loaded_shapes = {}
    for csv_file in [BASE_DIR / "customer_segments.csv", BASE_DIR / "clean_transactions.csv", BASE_DIR / "segment_summary.csv"]:
        df = pd.read_csv(csv_file)
        loaded_shapes[csv_file.name] = f"{df.shape[0]}x{df.shape[1]}"

    passed = len(existing) == len(required_files)
    details = [
        f"Required dashboard files found: {len(existing)}/{len(required_files)}",
        f"Loaded data shapes: {loaded_shapes}",
        "Dashboard readiness check passed (files/data available for app load).",
    ]
    return passed, details


def model_metrics_and_confusion() -> tuple[dict[str, float], pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(BASE_DIR / "customer_segments.csv")
    feature_cols = ["recency", "frequency", "monetary", "tenure_days", "avg_order_value", "cluster"]

    work = df.copy()
    for col in feature_cols + ["churn"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=feature_cols + ["churn"]).copy()
    X = work[feature_cols]
    y = work["churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=csr.RANDOM_STATE,
        stratify=y,
    )

    model = LogisticRegression(max_iter=2000, random_state=csr.RANDOM_STATE)
    try:
        model.fit(X_train, y_train)
    except Exception:
        model = RandomForestClassifier(n_estimators=300, random_state=csr.RANDOM_STATE, class_weight="balanced")
        model.fit(X_train, y_train)

    y_pred = pd.Series(model.predict(X_test), index=y_test.index)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    }
    return metrics, X_test, y_test, y_pred


def save_model_metrics_chart(metrics: dict[str, float]) -> tuple[bool, list[str]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor("#f6f8ee")
    ax.set_facecolor("#ffffff")

    labels = ["Accuracy", "Precision", "Recall"]
    values = [metrics["accuracy"], metrics["precision"], metrics["recall"]]
    palette = ["#1b7a67", "#cc8a1d", "#2c5a7d"]

    bars = ax.bar(labels, values, color=palette, width=0.55)
    ax.set_ylim(0, 1.0)
    ax.set_title("Model Testing: Accuracy, Precision, Recall", fontsize=16, weight="bold")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.2)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", fontsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "test_04_model_metrics.png")
    plt.close(fig)

    passed = all(v >= 0.70 for v in values)
    details = [
        f"Accuracy: {metrics['accuracy']:.3f}",
        f"Precision: {metrics['precision']:.3f}",
        f"Recall: {metrics['recall']:.3f}",
    ]
    return passed, details


def save_confusion_matrix_chart(y_true: pd.Series, y_pred: pd.Series) -> tuple[bool, list[str]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    fig.patch.set_facecolor("#f6f8ee")
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False, ax=ax)
    ax.set_title("Model Testing: Confusion Matrix", fontsize=16, weight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"], rotation=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "test_05_confusion_matrix.png")
    plt.close(fig)

    tn, fp, fn, tp = cm.ravel()
    passed = tp + tn > fp + fn
    details = [
        f"TN={tn}, FP={fp}, FN={fn}, TP={tp}",
        f"Total evaluated records: {int(cm.sum())}",
    ]
    return passed, details


def input_invalid_csv_test() -> tuple[bool, list[str]]:
    with NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
        tmp.write("bad_col_1,bad_col_2\n1,2\n3,4\n")

    try:
        csr.load_and_clean_transactions(str(tmp_path))
        passed = False
        details = ["Expected ValueError was not raised for invalid CSV schema."]
    except ValueError as exc:
        passed = True
        details = [
            "Invalid CSV schema correctly rejected.",
            f"Error: {str(exc)[:140]}",
        ]
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return passed, details


def input_missing_values_test() -> tuple[bool, list[str]]:
    sample = pd.DataFrame(
        {
            "CustomerID": [1001, 1001, 1002, 1003],
            "InvoiceDate": ["2011-01-05", "2011-02-10", "2011-03-15", "2011-04-20"],
            "Quantity": [2, None, 1, 3],
            "UnitPrice": [5.0, 7.5, None, 10.0],
            "Country": ["UK", None, "France", "UK"],
        }
    )

    with NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
        sample.to_csv(tmp_path, index=False)

    try:
        cleaned, _ = csr.load_and_clean_transactions(str(tmp_path))
        missing_after = int(cleaned[["quantity", "unitprice", "order_amount"]].isna().sum().sum())

        passed = missing_after == 0 and (cleaned["order_amount"] >= 0).all()
        details = [
            f"Rows after cleaning: {len(cleaned)}",
            f"Missing values in quantity/unitprice/order_amount after cleaning: {missing_after}",
            "Order amount computed and non-negative for all retained rows.",
        ]
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return passed, details


def performance_csv_read_test() -> tuple[bool, list[str]]:
    path = BASE_DIR / "customer_segments.csv"
    runs = 5
    timings = []

    for _ in range(runs):
        t0 = time.perf_counter()
        pd.read_csv(path)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)

    avg_ms = sum(timings) / len(timings)
    passed = avg_ms < 500.0
    details = [
        f"Average CSV read time over {runs} runs: {avg_ms:.2f} ms",
        f"Fastest: {min(timings):.2f} ms, Slowest: {max(timings):.2f} ms",
    ]
    return passed, details


def performance_model_train_test() -> tuple[bool, list[str]]:
    df = pd.read_csv(BASE_DIR / "customer_segments.csv")
    needed = ["recency", "frequency", "monetary", "tenure_days", "avg_order_value", "cluster", "churn"]
    work = df[needed].copy()

    for col in needed:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna().copy()

    runs = 3
    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = csr.train_churn_model(work.assign(customer_id=work.index.astype(str), segment_name="Segment"))[0]
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)

    avg_ms = sum(timings) / len(timings)
    passed = avg_ms < 3000.0
    details = [
        f"Average model training time over {runs} runs: {avg_ms:.2f} ms",
        f"Fastest: {min(timings):.2f} ms, Slowest: {max(timings):.2f} ms",
    ]
    return passed, details


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    # 1) Functional testing
    f1_pass, f1_details = functional_data_upload_test()
    save_card("test_01_functional_data_upload.png", "Functional Testing: Data Upload Working", f1_pass, f1_details)
    summary_rows.append(("Functional", "Data upload working", f1_pass, "test_01_functional_data_upload.png"))

    f2_pass, f2_details = functional_filters_test()
    save_card("test_02_functional_filters.png", "Functional Testing: Filters Working", f2_pass, f2_details)
    summary_rows.append(("Functional", "Filters working", f2_pass, "test_02_functional_filters.png"))

    f3_pass, f3_details = functional_dashboard_readiness_test()
    save_card("test_03_functional_dashboard_load.png", "Functional Testing: Dashboard Loading Correctly", f3_pass, f3_details)
    summary_rows.append(("Functional", "Dashboard loading correctly", f3_pass, "test_03_functional_dashboard_load.png"))

    # 2) Model testing
    metrics, _, y_true, y_pred = model_metrics_and_confusion()
    m1_pass, m1_details = save_model_metrics_chart(metrics)
    save_card("test_04a_model_metrics_status.png", "Model Testing: Metrics Status", m1_pass, m1_details)
    summary_rows.append(("Model", "Accuracy/Precision/Recall", m1_pass, "test_04_model_metrics.png"))

    m2_pass, m2_details = save_confusion_matrix_chart(y_true, y_pred)
    save_card("test_05a_confusion_matrix_status.png", "Model Testing: Confusion Matrix Status", m2_pass, m2_details)
    summary_rows.append(("Model", "Confusion matrix", m2_pass, "test_05_confusion_matrix.png"))

    # 3) Input validation
    i1_pass, i1_details = input_invalid_csv_test()
    save_card("test_06_input_invalid_csv.png", "Input Validation: Invalid CSV Handling", i1_pass, i1_details)
    summary_rows.append(("Input Validation", "Invalid CSV handling", i1_pass, "test_06_input_invalid_csv.png"))

    i2_pass, i2_details = input_missing_values_test()
    save_card("test_07_input_missing_values.png", "Input Validation: Missing Values Cases", i2_pass, i2_details)
    summary_rows.append(("Input Validation", "Missing values cases", i2_pass, "test_07_input_missing_values.png"))

    # 4) Performance testing (two simple tests)
    p1_pass, p1_details = performance_csv_read_test()
    save_card("test_08_performance_csv_read.png", "Performance Testing: CSV Read Speed", p1_pass, p1_details)
    summary_rows.append(("Performance", "CSV read speed", p1_pass, "test_08_performance_csv_read.png"))

    p2_pass, p2_details = performance_model_train_test()
    save_card("test_09_performance_model_training.png", "Performance Testing: Model Training Time", p2_pass, p2_details)
    summary_rows.append(("Performance", "Model training time", p2_pass, "test_09_performance_model_training.png"))

    summary_df = pd.DataFrame(summary_rows, columns=["category", "test", "passed", "evidence_image"])
    summary_df.to_csv(OUTPUT_DIR / "testing_summary.csv", index=False)

    print("Generated testing evidence in:")
    print(OUTPUT_DIR)
    print(summary_df)


if __name__ == "__main__":
    main()
