# Experiment 3: Handwritten Digit Classification – MLOps Pipeline

## 1. Aim

To implement an end-to-end MLOps pipeline for handwritten digit classification using the Scikit-learn Digits dataset and tools such as Git, DVC, Feast, and MLflow.

## 2. Dataset Information

The project uses the Scikit-learn Digits dataset.

- Total samples: 1,797
- Pixel features: 64
- Feature names: pixel_0_0 to pixel_7_7
- Target classes: 10 (digits 0 to 9)
- Dataset type: Handwritten digit images

## 3. Tools and Technologies

- Python
- Scikit-learn
- Pandas
- Git and GitHub
- DVC (Data Version Control)
- Feast (Feature Store)
- MLflow (Experiment Tracking)

## 4. Project Workflow

### Phase 1: Repository and Environment Setup
Create the project structure and configure `.gitignore` to exclude unnecessary files.

### Phase 2: Data Collection and DVC
- Load the Digits dataset using Scikit-learn.
- Save the dataset as a CSV file.
- Track the dataset using DVC.
- Configure local DVC remote storage.

### Phase 3: Feature Store using Feast
- Prepare the dataset in Parquet format.
- Add `image_id`, `event_timestamp`, and `created_timestamp`.
- Define the Entity, FileSource, FeatureView, and FeatureService.
- Register and materialize the features using Feast.

### Phase 4: MLflow Experiment Tracking
Train and compare two classification models:

1. Logistic Regression
2. Random Forest Classifier

Evaluate the models using:
- Accuracy
- Macro F1-score

Log model parameters, metrics, and model artifacts using MLflow.

### Phase 5: GitHub Integration
Use Git and GitHub to manage and share the project code and documentation.

## 5. Execution Commands

Run commands from the project root, adjusting paths if your team has organized the files differently.

**Data collection**
```bash
python src/collect.py
```

**DVC tracking**
```bash
dvc init
dvc add data/raw/digits_raw.csv
dvc push
```

**Feast**
```bash
python prepare_features.py
feast apply
```

Run Feast materialization using the command specified in the practical instructions.

**Model training**
```bash
python src/train.py
```

**MLflow UI**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open `http://localhost:5000` in a browser to view the experiments.

## 6. Expected Outputs

- Raw Digits dataset saved as a CSV file.
- Dataset tracked using DVC.
- Feature data prepared in Parquet format.
- Feast feature definitions registered and materialized.
- Logistic Regression and Random Forest models trained.
- Accuracy and Macro F1-score logged in MLflow.
- Experiment runs visible in the MLflow UI.

## 7. Conclusion

This experiment demonstrates an end-to-end MLOps workflow for handwritten digit classification. It combines data versioning, feature management, model training, experiment tracking, and version control using DVC, Feast, MLflow, Git, and GitHub.