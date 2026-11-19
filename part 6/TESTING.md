# Testing and Verification – Experiment 3

## 1. Objective

To verify the working of the handwritten digit classification MLOps pipeline and check the outputs of each phase.

## 2. Testing Checklist

### Phase 1: Repository and Environment
- [ ] Project folder structure is created.
- [ ] Python environment is configured.
- [ ] Required Python libraries are installed.
- [ ] `.gitignore` is configured.

### Phase 2: Data Collection and DVC
- [ ] `src/collect.py` executes successfully.
- [ ] `data/raw/digits_raw.csv` is generated.
- [ ] Dataset contains 1,797 samples.
- [ ] DVC initialization is successful.
- [ ] Dataset is tracked using DVC.
- [ ] DVC remote storage is configured.
- [ ] `dvc push` executes successfully.

### Phase 3: Feast Feature Store
- [ ] Feast repository is initialized.
- [ ] Feature preparation script executes successfully.
- [ ] Parquet feature file is generated.
- [ ] Entity and feature definitions are created.
- [ ] `feast apply` executes successfully.
- [ ] Feature materialization completes successfully.

### Phase 4: MLflow Experiment Tracking
- [ ] Logistic Regression model trains successfully.
- [ ] Random Forest model trains successfully.
- [ ] Accuracy is calculated.
- [ ] Macro F1-score is calculated.
- [ ] Model parameters and metrics are logged.
- [ ] MLflow UI opens successfully.
- [ ] Both experiment runs are visible in MLflow.

### Phase 5: GitHub Integration
- [ ] Project files are organized.
- [ ] Documentation is available.
- [ ] Changes are committed to Git.
- [ ] Team integration is reviewed.

## 3. Expected Results

The pipeline should collect the handwritten digit dataset, track it with DVC, prepare and register features using Feast, and train two classification models.

MLflow should display the experiment runs along with their parameters, accuracy, Macro F1-score, and model artifacts.

## 4. Testing Status

The checklist should be updated only after the corresponding step has been executed and verified.

## 5. Conclusion

Testing helps verify each stage of the MLOps pipeline and identify errors during data collection, feature management, model training, experiment tracking, and integration.