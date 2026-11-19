import mlflow 
import mlflow.sklearn 
import pandas as pd 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, f1_score 
 
mlflow.set_tracking_uri("sqlite:///mlflow.db") 
mlflow.set_experiment("digits-classification") 
 
df = pd.read_csv("data/raw/digits_raw.csv") 
X = df.drop("target", axis=1) 
y = df["target"] 
 
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.2, random_state=42, stratify=y 
) 
 
models = { 
    "logistic_regression": LogisticRegression(max_iter=5000, random_state=42), 
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42), 
} 
 
for name, model in models.items(): 
    with mlflow.start_run(run_name=name): 
        model.fit(X_train, y_train) 
        preds = model.predict(X_test) 
 
        acc = accuracy_score(y_test, preds) 
        f1 = f1_score(y_test, preds, average="macro") 
 
        mlflow.log_param("model_type", name) 
        mlflow.log_metric("accuracy", acc) 
        mlflow.log_metric("f1_macro", f1) 
 
        mlflow.sklearn.log_model(model, artifact_path="model") 
        print(f"{name} -> Accuracy: {acc:.4f}, F1 Macro: {f1:.4f}") 