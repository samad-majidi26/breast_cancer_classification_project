# Breast Cancer Classification: Comparing Machine Learning Models

A scikit-learn case study that predicts whether a breast tumor is **malignant or benign** from measurements of cell nuclei. The project compares seven classifiers, tests the effect of feature scaling, tunes hyperparameters with cross-validation, and checks for overfitting by comparing train and test scores.

> **Note:** This is a learning and portfolio project. It is not a medical tool and must not be used for clinical decisions.

## Dataset

- **Source:** the Breast Cancer Wisconsin (Diagnostic) dataset, loaded with `sklearn.datasets.load_breast_cancer`
- **Size:** 569 samples, 30 numeric features (mean, standard error and "worst" values of cell nucleus measurements such as radius, texture and smoothness)
- **Target:** binary (malignant / benign)
- **Data quality:** the notebook checks for missing values and duplicate rows before modeling

## Approach

1. **Data checks:** shape, missing values, duplicates, class balance
2. **Train/test split:** 80/20, stratified by class, `random_state=42`
3. **Baseline models on raw features:** K-Nearest Neighbors, Decision Tree, Logistic Regression, Random Forest, Gradient Boosting, SVC, XGBoost
4. **Baseline models on standardized features** (`StandardScaler`) to measure the effect of scaling
5. **Evaluation metrics:** accuracy, F1 score and ROC-AUC, each on train and test data to spot overfitting
6. **Hyperparameter tuning:** `GridSearchCV` with 5-fold cross-validation for all seven models
7. **Feature selection:** `SelectKBest` (top 6 features) and `SelectFromModel` with a Random Forest, followed by a final Random Forest on the selected features

## Results

Tuned models (5-fold `GridSearchCV` on the training set, evaluated on the held-out test set of 114 samples; features were not scaled for this step). F1 is macro-averaged.

| Model | CV accuracy | Train accuracy | Test accuracy | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9538 | 0.9604 | **0.9649** | **0.9623** | 0.9957 |
| SVC | 0.9560 | 0.9582 | 0.9561 | 0.9521 | **0.9964** |
| XGBoost | 0.9780 | 1.0000 | 0.9561 | 0.9521 | 0.9914 |
| KNN | 0.9429 | 1.0000 | 0.9474 | 0.9429 | 0.9734 |
| Random Forest | 0.9582 | 1.0000 | 0.9474 | 0.9435 | 0.9937 |
| Gradient Boosting | 0.9714 | 1.0000 | 0.9474 | 0.9429 | 0.9914 |
| Decision Tree | 0.9385 | 0.9912 | 0.9211 | 0.9164 | 0.9163 |

**Key findings**

- **Logistic Regression** had the best test accuracy (96.5%) and F1, and **SVC** the best ROC-AUC (0.996). Simple linear models matched or beat the more complex ensembles.
- **Overfitting:** XGBoost, KNN, Random Forest and Gradient Boosting reach 100% training accuracy, with train-test gaps of about 4-5 points. The linear models show almost no gap (Logistic Regression: -0.4 points).
- **Caution:** with only 114 test samples, one misclassified case changes accuracy by about 0.9 points, so the differences between the top models are small and not conclusive.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, Jupyter

## How to run

```bash
git clone https://github.com/samad-majidi26/breast_cancer_classification_project.git
cd breast_cancer_classification_project
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
jupyter notebook cancer_data.ipynb
```

The dataset ships with scikit-learn, so no download is needed.

## Project structure

```
├── cancer_data.ipynb   # full analysis: checks, models, tuning, feature selection
└── README.md
```

## Possible improvements

- Wrap scaling and models in a scikit-learn `Pipeline` so preprocessing is fitted on training data only, including inside cross-validation
- Optimize for recall on the malignant class, since missing a malignant case is the costly error
- Add a confusion matrix, ROC curves and feature importance plots
- Wrap the best model in a small Streamlit or FastAPI demo

## Author

Abdul Samad Majidi, Erfurt, Germany
