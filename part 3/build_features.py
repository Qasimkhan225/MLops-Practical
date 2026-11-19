from pathlib import Path
import subprocess
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [f"pixel_{i}" for i in range(64)]

def main():
    raw = pd.read_csv(ROOT / "data/raw/digits.csv")
    # Use a stable timestamp so reruns do not create artificial feature changes.
    raw["event_timestamp"] = pd.Timestamp("2024-01-01 00:00:00")
    out = ROOT / "data/feature_store/digit_features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    raw[["digit_id", "event_timestamp"] + FEATURES].to_parquet(out, index=False)

    subprocess.run(["feast", "apply"], cwd=ROOT / "feature_repo", check=True)
    print(f"Created Feast feature source with {len(raw)} records.")

if __name__ == "__main__":
    main()
