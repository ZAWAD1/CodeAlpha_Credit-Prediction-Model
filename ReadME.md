# 💳 Credit Scoring Model
## About

A machine learning pipeline that predicts whether a borrower will experience serious financial distress within two years. Trained on 150,000 real borrower records from the Give Me Some Credit dataset (Kaggle). Achieves ROC-AUC of 0.85+ using XGBoost with a Streamlit dashboard for live predictions.

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-green) ![Streamlit](https://img.shields.io/badge/Streamlit-red) ![scikit--learn](https://img.shields.io/badge/scikit--learn-orange)

---

## Project Structure

```
credit-scoring/
├── credit_dashboard.py           # Streamlit web app
├── credit_scoring.ipynb          # Full EDA + model training
├── Model/
│   └── credit_scoring_model.pkl  # Saved XGBoost model
├── data/
│   └── cs-training.csv           # Training dataset
└── README.md
```

---

## Dataset

**Give Me Some Credit** — Kaggle Competition Dataset

| Column | Plain English | Type |
|---|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Credit card utilization (0–1) | Float |
| `age` | Borrower age in years | Integer |
| `NumberOfTime30-59DaysPastDueNotWorse` | Times 30–59 days late | Integer |
| `DebtRatio` | Monthly debt ÷ monthly income | Float |
| `MonthlyIncome` | Self-reported monthly income ($) | Float |
| `NumberOfOpenCreditLinesAndLoans` | Total open credit accounts | Integer |
| `NumberOfTimes90DaysLate` | Times 90+ days late (most serious) | Integer |
| `NumberRealEstateLoansOrLines` | Mortgages / home equity loans | Integer |
| `NumberOfTime60-89DaysPastDueNotWorse` | Times 60–89 days late | Integer |
| `NumberOfDependents` | Financially dependent people | Integer |
| `SeriousDlqin2yrs` *(target)* | Will default in 2 years? (0/1) | Binary |

**Class distribution:** 93% no default · 7% default (imbalanced)

---

## Pipeline

### 1. Exploratory Data Analysis
- Class imbalance check — 93/7 split discovered
- Missing values — `MonthlyIncome` (20%), `NumberOfDependents` (3%)
- Outlier detection — `DebtRatio` up to 329,664; fake codes 96/98 in late payment columns
- Correlation heatmap — payment history features most correlated with default

### 2. Preprocessing
- Median imputation for missing values (robust to skewed income distribution)
- Outlier clipping at 99th percentile for income/debt; late payments capped at 10
- `StandardScaler` applied only to Logistic Regression inputs (tree models don't need scaling)

### 3. Feature Engineering

| Feature | Formula | Reason |
|---|---|---|
| `total_late` | sum of all 3 late payment columns | Single combined payment history signal — ranked #1 by XGBoost |
| `debt_amount` | `DebtRatio × MonthlyIncome` | Gives absolute debt context that ratio alone cannot |

### 4. Model Comparison

| Model | Precision | Recall | F1 (Default) | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.18 | 0.75 | 0.30 | 0.834 |
| Decision Tree (depth=5) | 0.19 | 0.80 | 0.31 | 0.853 |
| Random Forest | 0.57 | 0.15 | 0.24 | 0.839 |
| **XGBoost** ✅ | **0.28** | **0.63** | **0.39** | **0.852** |

**XGBoost won** due to sequential boosting, smarter class imbalance handling via `scale_pos_weight=9`, and built-in regularisation.

---

## Installation & Usage

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost streamlit joblib matplotlib seaborn
```

### 2. Train the model
```bash
jupyter notebook credit_scoring.ipynb
```
Run all cells top to bottom. Model saves to `Model/credit_scoring_model.pkl`.

### 3. Run the dashboard
```bash
streamlit run credit_dashboard.py
```
Opens at `http://localhost:8501`

---

## Key Findings

- **Past payment behaviour is the #1 predictor** — `total_late` ranked highest in feature importance, far above income or age
- **Accuracy is misleading on imbalanced data** — a model guessing "safe" for everyone scores 93%; ROC-AUC and F1 are the correct metrics
- **XGBoost's sequential learning outperforms voting** — each tree fixes the previous tree's mistakes rather than voting independently
- **A person with zero late payments predicts SAFE** regardless of income level, confirming what the feature importance shows

---

## Making a Prediction

-**Run the credit_dashboard.py to predict

```
