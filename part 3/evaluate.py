from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [f"pixel_{i}" for i in range(64)]

def main():
    test = pd.read_csv(ROOT / "data/processed/test.csv")
    model = joblib.load(ROOT / "models/best_model.joblib")
    X, y = test[FEATURES], test["target"]
    pred = model.predict(X)

    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "classes": list(range(10)),
    }

    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=4), encoding="utf-8")
    (report_dir / "classification_report.txt").write_text(
        classification_report(y, pred, digits=4, zero_division=0), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, display_labels=list(range(10)), cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title("Handwritten Digit Classification – Confusion Matrix")
    fig.tight_layout()
    fig.savefig(report_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(json.dumps(metrics, indent=4))
    print(f"Evaluation outputs saved in {report_dir}")

if __name__ == "__main__":
    main()
