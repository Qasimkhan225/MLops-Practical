from pathlib import Path
import pandas as pd
from sklearn.datasets import load_digits

ROOT = Path(__file__).resolve().parents[1]

def main():
    dataset = load_digits()
    feature_names = [f"pixel_{i}" for i in range(dataset.data.shape[1])]
    df = pd.DataFrame(dataset.data, columns=feature_names)
    df.insert(0, "digit_id", range(len(df)))
    df["target"] = dataset.target.astype(int)

    out = ROOT / "data/raw/digits.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} handwritten digit records to {out}")
    print(f"Number of pixel features: {len(feature_names)}")
    print("Target classes:", sorted(df["target"].unique().tolist()))

if __name__ == "__main__":
    main()
