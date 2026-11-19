from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [f"pixel_{i}" for i in range(64)]

def main():
    df = pd.read_csv(ROOT / "data/raw/digits.csv")
    X = df[FEATURES]
    y = df["target"]
    ids = df["digit_id"]

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURES, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURES, index=X_test.index
    )

    train = X_train_scaled.copy()
    train.insert(0, "digit_id", id_train.values)
    train["target"] = y_train.values

    test = X_test_scaled.copy()
    test.insert(0, "digit_id", id_test.values)
    test["target"] = y_test.values

    out_dir = ROOT / "data/processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    print(f"Training rows: {len(train)}, testing rows: {len(test)}")

if __name__ == "__main__":
    main()
