from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
OUTPUT = BASE_DIR.parent / "screenshots" / "report_images" / "model_validation_confusion_matrix.png"


def main() -> None:
    df = pd.read_csv(BASE_DIR / "customer_segments.csv")
    feature_cols = ["recency", "frequency", "monetary", "tenure_days", "avg_order_value", "cluster"]

    for col in feature_cols + ["churn"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=feature_cols + ["churn"]).copy()

    X = df[feature_cols]
    y = df["churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=160)
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix - Churn Model Validation")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"], rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
