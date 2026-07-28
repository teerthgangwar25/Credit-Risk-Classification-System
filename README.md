# Credit Risk Classification System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-gradient%20boosting-green)
![MLflow](https://img.shields.io/badge/MLflow-experiment%20tracking-0194E2)
![Pandas](https://img.shields.io/badge/Pandas-data%20processing-150458)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

An end-to-end machine learning pipeline that predicts credit default risk from applicant data, built around a business constraint rather than just a leaderboard metric: catch at least 75% of actual defaulters, not just score well on paper.

Four classifiers were built and compared, two of them tuned with cross-validated grid search, all six resulting runs tracked in MLflow. The final model is XGBoost at a deliberately chosen decision threshold, picked after the naive "best precision" threshold turned out to leave almost no safety margin against the actual requirement.

## Results

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.835 | 0.811 | 0.759 | 0.784 | 0.925 |
| Decision Tree | 0.825 | 0.775 | 0.785 | 0.780 | 0.818 |
| Random Forest (tuned) | 0.850 | 0.902 | 0.696 | 0.786 | 0.938 |
| **XGBoost (tuned)** | **0.875** | **0.855** | **0.823** | **0.839** | **0.953** |

**Random Forest posted the highest precision of any model here, and still doesn't make the cut.** Both its baseline and tuned versions fall short of the 75% recall requirement (70.9% and 69.6%), which disqualifies it for this use case regardless of its other numbers. XGBoost, deployed at a 0.45 classification threshold instead of the default 0.50, delivers 84.8% recall at 85.9% precision, the best F1 balance found across a full threshold sweep from 0.10 to 0.90.

One finding worth calling out: `Job`, an income and skill-level proxy, turned out to be the single strongest predictor for both tree-based models, ahead of loan amount and duration, despite not being examined directly during the original EDA.

## Project Structure

```
T2_Project_Teerth_Gangwar/
├── Notebooks/
│   ├── 01_EDA_Preprocessing.ipynb       # Data exploration, cleaning, feature engineering
│   ├── 02_Model_Development.ipynb       # 4 models, hyperparameter tuning, MLflow tracking
│   └── 03_Model_Evaluation.ipynb        # Metrics, visualizations, threshold analysis
├── Data/
│   ├── german_credit_data.csv           # Raw dataset (1,000 applications)
│   ├── X_train.csv / X_test.csv         # Processed feature splits
│   └── y_train.csv / y_test.csv         # Target splits
├── Models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl                      # Final recommended model
├── Report/
│   └── Technical_Report.pdf             # Full write-up: methodology, results, deployment plan
└── README.md
```

## Built With

- **Python** — pandas, numpy for data processing
- **scikit-learn** — Logistic Regression, Decision Tree, Random Forest, preprocessing pipelines
- **XGBoost** — gradient-boosted classifier, the final model
- **MLflow** — experiment tracking across all 6 training runs
- **matplotlib / seaborn** — EDA and evaluation visualizations
- **Google Colab** — development environment

## Methodology

1. **EDA and preprocessing** — missing-value handling (informative-missing, not random), one-hot encoding, stratified 80/20 split, post-split scaling to avoid leakage
2. **Feature engineering** — 4 engineered features, including `Account_Stability`, later validated by ranking in the top 3 most important features across every tree-based model
3. **Model development** — 4 classifiers built, Random Forest and XGBoost tuned via 5-fold `GridSearchCV` scored on F1
4. **Evaluation** — full metric comparison, ROC and precision-recall curves, feature importance, and an explicit threshold scan rather than assuming the default 0.5 cutoff is correct
5. **Business framing** — every modeling decision tied back to a real cost asymmetry: a missed defaulter costs far more than a wrongly-rejected good applicant, which is why recall, not accuracy, drives model selection here

Full reasoning, all visualizations, and the deployment plan are in [`Report/Technical_Report.pdf`](Report/Technical_Report.pdf).

## How to Run

All notebooks were built and tested in Google Colab.

1. Open a notebook in Colab and mount your Google Drive
2. Update `PROJECT_PATH` in the setup cell to point at your copy of this project folder
3. Run notebooks in order: `01` → `02` → `03` (each saves outputs the next one loads)

To run locally instead:

```bash
pip install pandas numpy scikit-learn xgboost mlflow matplotlib seaborn jupyter
jupyter notebook
```

Then replace the `drive.mount()` / `PROJECT_PATH` cell in each notebook with a local path.

## Dataset

German Credit Risk dataset, 1,000 loan applications across 10 features (age, job category, housing, account status, credit amount, duration, purpose), labeled as good or bad credit risk.

## Author

**Teerth Gangwar**
[GitHub](https://github.com/teerthgangwar25)
