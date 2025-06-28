# Project-Customer-Churn-Prediction-using-Machine-Learning

This project aims to predict customer churn in a financial institution using supervised machine learning algorithms. It involves a full pipeline from data preprocessing and exploratory analysis to feature engineering, model evaluation, and hyperparameter tuning.

## Dataset

The dataset contains 10,000 customer records with features including:

- Numerical: CreditScore, Age, Balance, EstimatedSalary
- Categorical: Geography, Gender, HasCrCard, IsActiveMember
- Target: `Exited` (1 if the customer left the bank, 0 otherwise)

## Libraries Used

- pandas, numpy
- seaborn, matplotlib
- scikit-learn
- xgboost
- lightgbm
- catboost

## Exploratory Data Analysis (EDA)

- Frequency analysis of churned vs. non-churned customers
- Distribution plots for numerical features (Age, Balance, CreditScore)
- Group-wise analysis based on categorical attributes (Gender, Geography)
- Visualizations: Histograms, Boxplots, Catplots, Heatmaps

## Data Preprocessing

- Handled missing values using:
  - Mode for categorical
  - Mean/median for numerical
- Feature scaling using robust scaler
- Outlier detection using IQR method
- One-hot encoding for categorical variables

## Feature Engineering

- Created new derived features:
  - `NewTenure` (Tenure / Age)
  - Binned scores for CreditScore, Age, Balance, and EstimatedSalary
- Dropped irrelevant columns: CustomerId, Surname

## Modeling and Evaluation

Evaluated multiple machine learning models using 10-fold cross-validation:

| Model                  | Accuracy (Mean ± Std) |
|------------------------|------------------------|
| Logistic Regression    | 81.11% ± 0.68%         |
| K-Nearest Neighbors    | 83.68% ± 1.09%         |
| Decision Tree (CART)   | 79.33% ± 1.41%         |
| Random Forest          | 86.41% ± 0.83%         |
| Support Vector Machine | 85.62% ± 0.95%         |
| Gradient Boosting      | 86.47% ± 0.87%         |
| LightGBM               | 86.54% ± 0.83%         |

Other evaluations included:

- Confusion matrix
- Classification report (precision, recall, F1-score)
- ROC-AUC curve
- Feature importance plots

## Model Tuning

Performed manual hyperparameter tuning on:

- LightGBM (learning_rate, max_depth, n_estimators, colsample_bytree)
- Gradient Boosting (learning_rate, max_depth, n_estimators, subsample)

## Key Findings

- LightGBM and Gradient Boosting yielded the highest prediction accuracy.
- Important features: Age, Balance, EstimatedSalary, Geography, and IsActiveMember.
- Active members are less likely to churn.
- Feature engineering and scaling significantly improved model performance.

## Future Work

- Implement full grid/randomized hyperparameter optimization
- Deploy the best model with Flask or Streamlit
- Add explainability using SHAP or LIME
- Handle class imbalance using SMOTE or similar techniques if required

## Installation

1. Clone the repository:

2. Install dependencies:

3. Run the notebook:
