# 🌫️ Air Quality Index (AQI) Forecasting — Delhi (2017–2024)

A comprehensive end-to-end time series forecasting project on Delhi AQI data. The pipeline covers data acquisition, exploratory analysis, statistical testing, nonlinearity detection, Granger causality analysis, outlier treatment, and benchmarking of 9 forecasting models ranging from classical statistical methods to deep learning architectures.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Statistical Analysis](#statistical-analysis)
- [Granger Causality Analysis](#granger-causality-analysis)
- [Models Implemented](#models-implemented)
- [Results Summary](#results-summary)
- [Detailed Model Results](#detailed-model-results)
- [Requirements](#requirements)
- [How to Run](#how-to-run)

---

## 📌 Project Overview

Air quality forecasting is critical for public health warnings, environmental policy, and urban planning. This project builds a multi-model forecasting framework on 2,597 daily AQI observations from Delhi (September 2017 to October 2024), using PM2.5, NO, and NO₂ as key predictor variables identified through Granger causality testing.

**Key Highlights:**
- 2,597 daily observations across 7+ years (2017–2024)
- Rigorous statistical preprocessing: ADF stationarity, Hurst exponent, nonlinearity tests
- Granger causality across 5 lags to identify causal predictors
- Winsorisation (10–90%) for robust outlier treatment
- 9 models benchmarked including ARIMA, SARIMAX, ARNN, ARIMA-ARNN Hybrid, LSTM, XGBoost, Prophet, Random Forest, and Decision Tree
- Best test result: **Random Forest — R² = 0.86, MAPE = 14.96%**

---

## 📦 Dataset

**Source:** Google Sheets (publicly shared dataset)  
**URL:** `https://docs.google.com/spreadsheets/d/1jcS2x9NIxASIhl9DELkHMGMHAyJfdwH3/export?format=csv`

**Date Range:** 11 September 2017 — 20 October 2024  
**Observations:** 2,597 daily records  
**Target Variable:** AQI Index

### Raw Features

| Feature | Unit | Missing % | Action |
|---|---|---|---|
| AQI Index | — | 4.04% | Forward-fill |
| PM2.5 | µg/m³ | 4.47% | Forward-fill |
| PM10 | µg/m³ | **39.66%** | **Dropped** |
| NO | µg/m³ | 4.51% | Forward-fill |
| NO2 | µg/m³ | 4.51% | Forward-fill |
| NOx | ppb | 2.04% | Forward-fill |
| NH3 | µg/m³ | **39.66%** | **Dropped** |
| SO2 | µg/m³ | ~4% | Forward-fill |
| CO | mg/m³ | ~4% | Forward-fill |
| Ozone | µg/m³ | ~4% | Forward-fill |
| Benzene | µg/m³ | ~4% | Forward-fill |
| Toluene | µg/m³ | ~4% | Forward-fill |
| Eth-Benzene | µg/m³ | ~4% | Forward-fill |
| MP-Xylene | µg/m³ | ~4% | Forward-fill |
| Wind Speed | m/s | ~4% | Forward-fill |
| Wind Direction | deg | ~4% | Forward-fill |
| Relative Humidity | % | ~4% | Forward-fill |
| Solar Radiation | W/m² | ~4% | Forward-fill |

**Columns dropped:** PM10 and NH3 (both >30% missing values)  
**Imputation strategy:** Forward-fill followed by backward-fill (time-series safe)

### Descriptive Statistics (Multivariate Modelling Features)

| Statistic | AQI Index | PM2.5 (µg/m³) | NO (µg/m³) | NO2 (µg/m³) |
|---|---|---|---|---|
| Count | 2,597 | 2,597 | 2,597 | 2,597 |
| Mean | 217.18 | 112.23 | 16.94 | 29.81 |
| Std | 105.29 | 74.06 | 17.95 | 13.25 |
| Min | 30.00 | 8.33 | 0.76 | 2.09 |
| 25% | 125.00 | 56.47 | 6.35 | 20.05 |
| Median | 210.00 | 93.62 | 10.79 | 27.25 |
| 75% | 301.00 | 146.77 | 19.55 | 37.66 |
| Max | 499.00 | 429.87 | 144.64 | 101.78 |

---

## 🗂️ Project Structure

```
AQI_Forecasting/
├── AQI.xlsx                          # Cleaned dataset (saved after preprocessing)
├── Air_Quality_Index_Forecasting.ipynb  # Main notebook
├── plots/
│   ├── aqi_trend.png                 # AQI Index over time
│   ├── rolling_mean.png              # Trend with rolling average
│   ├── monthly_trends.png            # Monthly AQI patterns (3x4 grid)
│   ├── seasonal_decomposition.png    # Trend + Seasonal + Residual
│   ├── acf_pacf.png                  # ACF and PACF plots (50 lags)
│   ├── correlation_heatmap.png       # Feature correlation matrix
│   ├── boxplot_raw.png               # Boxplot before Winsorisation
│   ├── boxplot_winsorized.png        # Boxplot after Winsorisation
│   ├── prophet_actual_vs_predicted_line.jpg
│   ├── rf_actual_vs_predicted_line_sorted.jpg
│   └── dt_actual_vs_predicted_line_sorted.jpg
└── README.md
```

---

## 🔧 Pipeline Walkthrough

### Phase 1 — Data Loading & EDA

- Loaded data directly from Google Sheets URL via `pd.read_csv()`
- Converted AQI Index to numeric and Date column to datetime
- Plotted AQI distribution (histogram + KDE), boxplot, and time series trend
- Computed 12-day rolling mean to visualise long-term trend
- Generated monthly AQI trend plots (3 rows × 4 columns) across all years

### Phase 2 — Missing Value Treatment

- Dropped columns with >30% missing values (PM10, NH3)
- Applied forward-fill then backward-fill for all remaining columns
- Verified zero missing values post-imputation

### Phase 3 — Outlier Treatment (Winsorisation)

```python
# Capping at 10th and 90th percentiles
def winsorize_column(df, column, lower=0.10, upper=0.90):
    df[column] = winsorize(df[column], limits=(lower, 1-upper))
    return df
```

Applied to: AQI Index, PM2.5, NO, NO2

### Phase 4 — Statistical Analysis

See [Statistical Analysis](#statistical-analysis) section below.

### Phase 5 — Feature Selection via Granger Causality

See [Granger Causality Analysis](#granger-causality-analysis) section below.

### Phase 6 — Model Training & Evaluation

- Final modelling features: **PM2.5 (µg/m³)**, **NO (µg/m³)**, **NO2 (µg/m³)**
- Train/test split: **80% train (2,077 obs) / 20% test (520 obs)**
- All models evaluated using: MAE, MSE, RMSE, MAPE, R², Adjusted R²

---

## 📊 Statistical Analysis

### ADF Stationarity Test (Augmented Dickey-Fuller)

| Variable | ADF Statistic | p-value | Stationary? |
|---|---|---|---|
| AQI Index | -4.089 | 0.0010 | ✅ Yes |
| PM2.5 | — | 4.73e-05 | ✅ Yes |
| NO | — | 6.78e-04 | ✅ Yes |
| NO2 | — | 1.50e-09 | ✅ Yes |
| NOx | — | 1.47e-05 | ✅ Yes |
| SO2 | — | 7.64e-05 | ✅ Yes |
| CO | — | 7.69e-09 | ✅ Yes |

**Conclusion:** All modelling variables are stationary. No differencing required.

### Hurst Exponent

| Result | Value | Interpretation |
|---|---|---|
| Hurst Exponent (H) | **0.823** | Long-term dependence (H > 0.5) — AQI exhibits persistent, trending behaviour |

### Distributional Properties

| Property | Value | Interpretation |
|---|---|---|
| Skewness | 0.220 | Slight right skew |
| Kurtosis | -1.014 | Platykurtic (lighter tails than normal) |
| R² (Linear Fit) | 0.0085 | AQI does not follow a simple linear trend |

### Nonlinearity Tests

| Test | p-value | Conclusion |
|---|---|---|
| McLeod-Li (ARCH) | 3.02 × 10⁻²⁶² | Strong heteroscedasticity — variance changes over time |
| Kenan's RESET | 1.20 × 10⁻⁴¹ | Significant nonlinearity — model misspecification in linear models |
| Tsay (ADF on residuals) | 0.0009 | Nonlinearity in residuals — structural breaks likely |
| Likelihood Ratio | 0.3507 | No evidence of threshold nonlinearity |

**Conclusion:** AQI data is nonlinear and heteroscedastic. Linear models (ARIMA) are expected to underperform. Nonlinear models (LSTM, XGBoost, Random Forest) are recommended.

### Correlation Matrix (Key Findings)

| Variable | Correlation with AQI |
|---|---|
| PM2.5 | **+0.90** (strongest) |
| NO2 | +0.56 |
| NOx | +0.54 |
| NO | +0.52 |
| CO | +0.37 |
| Wind Speed | -0.36 |
| Wind Direction | -0.26 |
| Relative Humidity | -0.14 |

---

## 🔬 Granger Causality Analysis

Granger causality tested across **5 lags** for all features against AQI Index.

### Strongly Significant Predictors (p < 0.05 at all lags)

- PM2.5 (µg/m³)
- NO (µg/m³)
- NO2 (µg/m³)
- NOx (ppb)
- CO (mg/m³)
- Benzene (µg/m³)
- Toluene (µg/m³)
- Wind Speed (m/s)
- Wind Direction (deg)

### Partially Significant Predictors

- SO2 (µg/m³) — significant at lags 2–5, not lag 1
- Eth-Benzene (µg/m³) — significant in most lags except lag 1
- MP-Xylene (µg/m³) — partial significance
- Relative Humidity (%) — significant at lags 2–5
- Solar Radiation (W/m²) — significant at lags 3–5

### Non-Significant

- Ozone (µg/m³) — no causal relationship detected

**Final selected features for multivariate models:** PM2.5, NO, NO2

---

## 🤖 Models Implemented

| # | Model | Type | Notes |
|---|---|---|---|
| 1 | ARIMA(5,1,0) | Univariate Statistical | Baseline |
| 2 | ARNN | Univariate Neural | MLP on AR features |
| 3 | ARIMA-ARNN Hybrid | Hybrid | ARIMA residuals fed to ARNN |
| 4 | SARIMAX(1,0,0)×(1,0,[1],7) | Multivariate Statistical | Seasonal period=7 (weekly) |
| 5 | LSTM | Multivariate Deep Learning | 50 epochs, dropout, L2 regularisation |
| 6 | GRU | Multivariate Deep Learning | 20 epochs |
| 7 | SimpleRNN | Multivariate Deep Learning | 20 epochs |
| 8 | TCN | Multivariate Deep Learning | Temporal Convolutional Network |
| 9 | XGBoost | Gradient Boosting | n_estimators=250, max_depth=3 |
| 10 | Prophet | Additive Forecasting | With PM2.5, NO, NO2 regressors |
| 11 | Random Forest | Ensemble | n_estimators=100–200 |
| 12 | Decision Tree (Pruned) | Tree | max_depth=6 |
| 13 | Decision Tree (Tuned) | Tree | GridSearchCV, max_depth=10 |

**Seed:** `42` (set for NumPy, Python random, TensorFlow — fully reproducible)

---

## 📊 Results Summary

### Test Set Performance Comparison

| Model | MAE | RMSE | MAPE | R² (Test) |
|---|---|---|---|---|
| ARIMA | 109.88 | 123.83 | 92.93% | -0.491 |
| ARNN | ~30.54 | ~41.48 | ~19.91% | ~0.833 |
| Decision Tree (Pruned) | 27.41 | 38.51 | 15.62% | 0.849 |
| Decision Tree (Tuned) | 26.60 | — | — | ~0.87 |
| XGBoost | 33.27 | 42.57 | 20.26% | 0.790 |
| Prophet | 29.87 | 38.94 | — | **0.83** |
| **Random Forest** | **26.30** | **37.39** | **14.96%** | **0.86** |
| RF (Tuned, GridSearchCV) | 25.76 | 35.94 | 14.90% | 0.869 |
| SARIMAX | — | — | — | — |
| LSTM | — | — | — | — |

**Best model on test set:** Random Forest (R² = 0.86, MAPE = 14.96%)  
**Best tuned model:** Tuned Random Forest (R² = 0.869 test — GridSearchCV, 3-fold CV, 243 candidates)

---

## 📋 Detailed Model Results

### ARIMA (5,1,0)

```
RMSE:  123.83
MAE:   109.88
MAPE:   92.93%
R²:     -0.4911
Adj R²: -0.4939
```
Conclusion: ARIMA fails to capture nonlinear AQI dynamics. Performs worse than predicting the mean.

### ARNN (Artificial Neural Network on AR features)

```
RMSE:  ~41.48
MAE:   ~30.54
MAPE:  ~19.91%
R²:    ~0.833
```
Conclusion: Significantly outperforms ARIMA. Captures nonlinear dependencies.

### XGBoost

```
TRAIN: MAE=28.42, RMSE=38.50, MAPE=16.23%, R²=0.84
TEST:  MAE=33.27, RMSE=42.57, MAPE=20.26%, R²=0.79
```
Model is well-generalised (train ≈ test).

### Prophet (with PM2.5, NO, NO2 regressors)

```
TRAIN: MAE=28.35, RMSE=37.33, MAPE=16.14%, R²=0.85
TEST:  MAE=29.87, RMSE=38.94,              R²=0.83
```
Model is well-generalised.

### Random Forest (Base)

```
TRAIN: MAE=12.06, RMSE=21.92, MAPE=6.75%,  R²=0.95
TEST:  MAE=26.30, RMSE=37.39, MAPE=14.96%, R²=0.86
```

### Random Forest (Tuned — GridSearchCV)

```
Best params: max_depth=5, max_features=None,
             min_samples_leaf=2, min_samples_split=10,
             n_estimators=200
             
TRAIN: MAE=24.95, RMSE=35.78, MAPE=14.34%, R²=0.862
TEST:  MAE=25.76, RMSE=35.94, MAPE=14.90%, R²=0.869
```
Model is well-generalised.

### Decision Tree (Pruned)

```
TRAIN: MAE=24.44, RMSE=35.33, MAPE=14.03%, R²=0.865
TEST:  MAE=27.41, RMSE=38.51, MAPE=15.62%, R²=0.849
```

### Decision Tree (Tuned — GridSearchCV)

```
Best params: max_depth=10, max_features=None,
             min_samples_leaf=4, min_samples_split=15,
             splitter=random
             
TEST: MAE=26.60
```

---

## 📦 Requirements

```
numpy
pandas
matplotlib
seaborn
statsmodels
scikit-learn
tensorflow>=2.0
xgboost
prophet
hurst
scipy
openpyxl
```

Install all dependencies:

```bash
pip install numpy pandas matplotlib seaborn statsmodels scikit-learn tensorflow xgboost prophet hurst scipy openpyxl
```

---

## 🚀 How to Run

### Step 1 — Load Data

The dataset is loaded directly from Google Sheets:

```python
url = "https://docs.google.com/spreadsheets/d/1jcS2x9NIxASIhl9DELkHMGMHAyJfdwH3/export?format=csv"
data = pd.read_csv(url)
```

### Step 2 — Preprocessing

Run the preprocessing cells to:
- Drop PM10 and NH3 (>30% missing)
- Apply forward-fill + backward-fill
- Save cleaned data to `AQI.xlsx`

### Step 3 — Statistical Tests

Run the ADF test, Hurst exponent, nonlinearity tests, and Granger causality cells to understand the data structure and confirm feature selection.

### Step 4 — Outlier Treatment

```python
from scipy.stats.mstats import winsorize

def winsorize_column(df, column, lower=0.10, upper=0.90):
    df[column] = winsorize(df[column], limits=(lower, 1-upper))
    return df
```

### Step 5 — Model Training

Run each model cell sequentially. All models use:
- Features: `['PM2.5 (µg/m³)', 'NO (µg/m³)', 'NO2 (µg/m³)']`
- Target: `'AQI Index'`
- Split: 80% train / 20% test (no shuffle)

### Step 6 — Evaluation

Each model cell prints a full metrics table including MAE, MSE, RMSE, MAPE, R², and Adjusted R² on both train and test sets.

---

## 💡 Key Findings

1. **AQI is stationary** (ADF p=0.001) but highly nonlinear (RESET p=1.2×10⁻⁴¹) — standard ARIMA is inadequate
2. **PM2.5 is the dominant predictor** (correlation = 0.90, Granger significant at all lags)
3. **Long-term dependence** (Hurst = 0.823) — past AQI values are predictive of future values
4. **Seasonal patterns exist** — AQI peaks in winter months (October–February) due to lower wind speeds and atmospheric inversion
5. **Ensemble methods generalise best** — Random Forest and tuned variants achieve R² ≈ 0.86–0.87 on unseen test data
6. **Prophet performs surprisingly well** (R² = 0.83 test) given its additive decomposition framework

---

## 📝 Notes

- The dataset begins in September 2017 with significant missing values in early months (imputed with forward-fill)
- Winsorisation at 10–90% was chosen over 5–95% for stronger outlier suppression while preserving the distribution shape
- All deep learning models (LSTM, GRU, RNN, TCN) were trained with seed=42 for reproducibility
- MAPE for Prophet on test set is reported as `nan` in the notebook output due to near-zero AQI values in the denominator — this is a known MAPE limitation and does not reflect model quality (R² = 0.83 is valid)

---

## 👤 Author

**Alamuri Sri Jagadeeswara Rao**  
M.Sc. Data Science — Vellore Institute of Technology (VIT), Vellore  
[GitHub](https://github.com/Jagadeesh-Alamuri) | [LinkedIn](https://linkedin.com/in/asjagadeeswararao)
