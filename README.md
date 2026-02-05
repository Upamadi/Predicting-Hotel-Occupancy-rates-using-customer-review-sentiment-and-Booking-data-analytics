# Predicting-Hotel-Occupancy-rates-using-customer-review-sentiment-and-Booking-data-analytics
Predictive analytics pipeline to forecast hotel occupancy by combining operational performance metrics with customer sentiment derived from online reviews. The goal is to test whether behavioural indicators (sentiment) extracted from guest reviews improve forecasting accuracy and provide actionable insights for hotel management.

---

## Table of contents

- [Project overview](#project-overview)
- [Objectives](#objectives)
- [Dataset description](#dataset-description)
  - [Operational dataset](#operational-dataset)
  - [Customer review dataset](#customer-review-dataset)
- [Methodological pipeline](#methodological-pipeline)
- [Feature engineering](#feature-engineering)
- [Predictive modelling & evaluation](#predictive-modelling--evaluation)
- [Reproducibility & usage](#reproducibility--usage)
  - [Requirements](#requirements)
  - [Running the pipeline](#running-the-pipeline)
- [Project structure](#project-structure)
- [Results & interpretation](#results--interpretation)
- [Limitations & next steps](#limitations--next-steps)
- [License](#license)
- [Contact](#contact)

---

## Project overview

This project builds an end-to-end pipeline that:

- Validates and prepares synthetic operational hotel data and guest review text,
- Converts unstructured review text to sentiment indicators (VADER),
- Constructs temporal features (including lags) to capture delayed behavioural effects,
- Trains regression-based forecasting models to predict monthly hotel occupancy rate,
- Produces visualizations and an optional dashboard for managerial interpretation.

The pipeline is intentionally modular so that real-world data can replace synthetic inputs with minimal changes.

---

## Objectives

- Construct a clean, analysable dataset aligned on hotel-month observations.
- Transform textual reviews to numeric sentiment features and aggregate to hotel-month level.
- Engineer temporal features (lags, rolling statistics) to capture inertia and delayed impacts.
- Compare models with and without sentiment indicators to quantify their value-add.
- Provide reproducible scripts, notebooks, and visual artifacts to support managerial decision-making.

---

## Dataset description

### Operational dataset
- Synthetic dataset simulating hotel-month observations.
- Key columns:
  - `hotel_id` — unique hotel identifier
  - `year`, `month` — temporal keys
  - `occupancy_rate` — target variable (0–100 % or 0–1 depending on representation)
  - `adr` — Average Daily Rate
  - `cancellation_rate` — cancellations as a proportion
  - other engineered operational indicators
- Generation uses domain-informed constraints to maintain realistic relationships (e.g., occupancy ↔ ADR).

### Customer review dataset
- Contains guest review text aggregated by `hotel_id`, `year`, `month`.
- Sentiment computed per review and aggregated (mean, median, share positive/negative).
- Linked to operational dataset via hotel-month keys.

Note: Data included in this repository are synthetic and intended for demonstration and reproducibility.

---

## Methodological pipeline

1. Data validation
   - Schema checks, null-handling, and range assertions to ensure analytic reliability.
2. Text processing & sentiment
   - Preprocessing (lowercasing, basic token cleaning) and VADER sentiment scoring for each review.
   - Aggregate sentiment to hotel-month: mean compound score, std, proportion positive/negative.
3. Feature engineering
   - Temporal features: month-of-year (seasonality), lagged occupancy/sentiment (e.g., 1-3 month lags), rolling means.
   - Interaction features where relevant (e.g., sentiment × ADR).
4. Dataset integration
   - Merge operational and aggregated sentiment on `hotel_id`, `year`, `month`.
   - Final validation for alignment and missingness.
5. Modelling & evaluation
   - Baseline models: OLS / regularized linear models (Ridge, Lasso).
   - Tree-based models: Random Forest, Gradient Boosting (e.g., XGBoost/LightGBM).
   - Cross-validation grouped by hotel (time-series aware CV) and temporal holdout for out-of-sample forecasting.
   - Metrics: RMSE, MAE, R², and business-oriented metrics (e.g., occupancy bins accuracy).
6. Visualization & dashboard
   - Exploratory plots, feature importances, partial dependence plots.
   - Optional interactive dashboard (Streamlit / Dash) to inspect forecasts and drivers.

---

## Feature engineering

Common features created by the pipeline:

- Sentiment aggregates: `sentiment_mean`, `sentiment_std`, `sentiment_pos_share`, `sentiment_neg_share`.
- Lagged sentiment: `sentiment_mean_lag_1`, `sentiment_mean_lag_3`.
- Operational lags: `occupancy_lag_1`, `occupancy_lag_3`.
- Rolling statistics: 3-month and 6-month rolling means for occupancy and ADR.
- Seasonality encodings: `month`, `is_peak_season`, or cyclical transforms (`sin_month`, `cos_month`).

---

## Predictive modelling & evaluation

- Baseline comparison: models trained with operational features only vs. operational + sentiment features.
- Cross-validated evaluation with a time-aware split (train on earlier months, validate on later months).
- Feature importance used to assess the contribution of sentiment features.
- Example metrics reported:
  - RMSE reduction when sentiment is included.
  - Change in explained variance (ΔR²).
  - Business impact (e.g., reduced forecasting error during promotions or high-cancellation periods).

---

## Reproducibility & usage

### Requirements
- Python 3.9+ recommended
- Common packages (example): pandas, numpy, scikit-learn, nltk, vaderSentiment, matplotlib, seaborn, xgboost or lightgbm, streamlit (if using dashboard)
- Install via pip:
  - pip install -r requirements.txt
  - Or create a conda env: conda env create -f environment.yml (if provided)

### Running the pipeline
1. Install dependencies.
2. Generate or place datasets:
   - The repository contains a synthetic generator script: `scripts/generate_synthetic_data.py` (if present).
   - Or place `operational.csv` and `reviews.csv` under `data/raw/`.
3. Preprocess and compute sentiment:
   - python scripts/compute_sentiment.py --reviews data/raw/reviews.csv --out data/processed/sentiment_agg.csv
4. Build features / merge:
   - python scripts/build_features.py --ops data/raw/operational.csv --sent data/processed/sentiment_agg.csv --out data/processed/features.csv
5. Train & evaluate models:
   - python scripts/train_models.py --features data/processed/features.csv --out results/
6. (Optional) Launch dashboard:
   - streamlit run dashboard/app.py -- --data results/forecasts.csv

Adjust script names/flags to the files present in the repository.

---

## Project structure (suggested)

- data/
  - raw/ (operational.csv, reviews.csv or synthetic generator output)
  - processed/ (merged features, aggregates)
- notebooks/
  - 01_data_validation.ipynb
  - 02_sentiment_processing.ipynb
  - 03_feature_engineering_eda.ipynb
  - 04_modeling_and_evaluation.ipynb
- scripts/
  - generate_synthetic_data.py
  - compute_sentiment.py
  - build_features.py
  - train_models.py
- dashboard/
  - app.py (Streamlit/Dash)
- results/
  - figures/, models/, forecasts.csv
- requirements.txt
- README.md

---

## Results & interpretation

- The repository contains example outputs and visualizations in `results/` that illustrate:
  - Occupancy forecasts vs actuals
  - Feature importance rankings
  - How sentiment features change model performance (quantified by RMSE / R²)
- Summary finding (example): including lagged aggregated sentiment improved RMSE by X% on the held-out period and increased explained variance, particularly during months with significant guest feedback shifts (promotions, service changes).

(Replace the above with precise numbers once models are run.)

---

## Limitations & next steps

- Synthetic data: while useful for pipeline demonstration, results may not generalize until tested on real-world aligned datasets.
- Sentiment tool: VADER is tuned for social media and short reviews—consider fine-tuned transformer classifiers for deeper semantic nuance.
- Causality: correlations observed do not imply causation. Consider quasi-experimental designs for causal inference.
- Additional improvements:
  - Enrich with booking lead-time, channel mix, competitor pricing.
  - Use hierarchical models to pool information across hotels.
  - Integrate more advanced NLP (topic modeling, aspect-based sentiment).

---

## License

This project is released under the MIT License. See LICENSE for details.

---

## Contact

Maintainer: Upamadi  
For questions or contributions, open an issue or submit a pull request.

Acknowledgements:
- Sentiment analysis via VADER (Hutto & Gilbert).
- Example visualizations adapted from common forecasting best practices.
