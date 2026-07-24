# House Price Prediction Using Advanced Regression Techniques

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red?style=flat-square)
![Status](https://img.shields.io/badge/Status-90%25%20Complete-yellow?style=flat-square)

</div>

## Project Overview

This project predicts house prices using advanced regression and machine learning techniques. It uses the Kaggle **House Prices: Advanced Regression Techniques** dataset, also known as the Ames Housing dataset.

The goal of this project is to analyze housing data, understand important factors affecting property prices, build multiple regression models, compare their performance, and optionally deploy the best model using a Streamlit web application.

**The project includes:**

- Research report on Data Analytics, Data Science, AI, and advanced regression
- Action plan with project methodology and timeline
- Exploratory Data Analysis and hypothesis testing
- Data preprocessing and feature engineering
- Advanced regression model implementation
- Feature selection
- Hyperparameter tuning
- Model evaluation and comparison
- Streamlit web application for house price prediction

---

## Dataset

**Source:** [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

Files: `train.csv`, `test.csv`, `data_description.txt`

Target variable: `SalePrice` (right-skewed, so log-transformed during training via `np.log1p` and converted back with `np.expm1`).

---

## Project Structure

```text
House_Price_Prediction/
│
├── Module_1_Research_Report/
│   └── research_report.pdf
│
├── Module_2_Action_Plan/
│   └── action_plan.pdf
│
├── Module_3_EDA/
│   ├── eda_analysis.ipynb
│   └── visualizations/
│
├── Module_4_Model_Building/
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── model_building.ipynb
│
├── Module_5_Web_App/
│   ├── app.py
│   ├── requirements.txt
│   └── artifacts/
│       ├── best_model.pkl
│       ├── scaler.pkl
│       ├── feature_columns.pkl
│       ├── default_values.pkl
│       ├── best_model_name.pkl
│       └── model_results.csv
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── data_description.txt
│   └── processed/
│
├── README.md
└── requirements.txt
```
---

## Exploratory Data Analysis & Hypothesis Testing

EDA covers missing values, distributions, correlations, outlier detection, and neighborhood price comparisons, visualized with histograms, Q-Q plots, scatter plots, box plots, and correlation heatmaps.

| Hypothesis | Test | Result |
|---|---|---|
| `GrLivArea` correlates with `SalePrice` | Pearson correlation | Rejected H0 — r ≈ 0.71, p < 0.001 |
| `SalePrice` differs by neighborhood | One-way ANOVA | Rejected H0 — F ≈ 71.78, p < 0.001 |
| `HouseAge` correlates (negatively) with `SalePrice` | Pearson correlation | Rejected H0 — r ≈ -0.52, p < 0.001 |

Two `GrLivArea` outliers (Id 524, Id 1299) are removed before training.

## Feature Engineering

```text
HouseAge   = YrSold - YearBuilt
RemodAge   = YrSold - YearRemodAdd
TotalSF    = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
TotalBath  = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath
```

Feature selection combines Lasso coefficients, Random Forest / XGBoost feature importance, and Recursive Feature Elimination. Top predictors: `OverallQual`, `GrLivArea`, `TotalSF`, `GarageCars`, `GarageArea`, `TotalBsmtSF`, `1stFlrSF`, `YearBuilt`, `HouseAge`, and neighborhood-related features.

## Models

| Model | Notes |
|---|---|
| Polynomial Regression | Captures non-linear relationships on selected predictors |
| Ridge Regression | L2 regularization, handles multicollinearity |
| Lasso Regression | L1 regularization, doubles as feature selector |
| Random Forest | Ensemble model, captures feature interactions |
| XGBoost | Gradient boosting, strong baseline for tabular data |

Hyperparameters for Ridge, Lasso, Random Forest, and XGBoost are tuned via `GridSearchCV` / `RandomizedSearchCV`. Models are compared using MSE, RMSE, MAE, and R², with RMSE (in original dollar scale) as the primary metric. Results are saved to `Module_5_Web_App/artifacts/model_results.csv` after running the model-building notebook; the lowest-RMSE model is selected for deployment.

## Key Findings

- `OverallQual` and `GrLivArea` are the strongest predictors of sale price.
- Neighborhood has a statistically significant effect on price.
- Newer homes and larger garages/basements correlate with higher prices.
- Some missing values are meaningful (e.g., no pool, garage, basement, or alley) rather than data gaps.

## Streamlit Web App

`Module_5_Web_App/app.py` loads the saved model artifacts and lets users enter features (overall quality, living area, garage capacity, total square footage, year built, sale year, bathrooms, neighborhood) to get a predicted sale price. Unspecified fields default to the training data's median/mode values.
---

---
## Quick Start

**Want to try the app immediately?**

```bash
# 1. Clone and setup
git clone https://github.com/Priyal9497/House_Price_Prediction.git
cd House_Price_Prediction
pip install -r requirements.txt

# 2. Download dataset to data/raw/
# Place train.csv and test.csv inside data/raw/

# 3. Run model training
jupyter notebook Module_4_Model_Building/model_building.ipynb

# 4. Launch web app
cd Module_5_Web_App
streamlit run app.py
```
---

## Installation and Setup

```bash
# Clone and install
git clone https://github.com/Priyal9497/House_Price_Prediction.git
cd House_Price_Prediction
pip install -r requirements.txt

# Add train.csv and test.csv to data/raw/

# Train models (also generates artifacts for the web app)
jupyter notebook Module_4_Model_Building/model_building.ipynb

# Run the web app
cd Module_5_Web_App
streamlit run app.py
```

**Requirements:** pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, scipy, streamlit, joblib, jupyter, ipykernel, plotly, statsmodels

---

## Example Web App Input

Example values:

```text
Overall Quality: 7
Above Ground Living Area: 1800
Garage Capacity: 2
Total Square Footage: 2500
Year Built: 2005
Sale Year: 2010
Full Bathrooms: 2
Neighborhood: CollgCr
```

The app returns an estimated house sale price.

---

## Troubleshooting

- **`best_model.pkl` not found** — run all cells in `model_building.ipynb`; it must generate the full `artifacts/` folder (`best_model.pkl`, `scaler.pkl`, `feature_columns.pkl`, `default_values.pkl`, `best_model_name.pkl`, `model_results.csv`).
- **Feature mismatch in the app** — ensure `feature_columns.pkl`, `default_values.pkl`, and the scaler all come from the same training run, and that input columns match training order.

## Limitations & Ethical Considerations

This model is trained on historical Ames, Iowa data and does not account for current market conditions, inflation, interest rates, or demand. It is intended for educational purposes and should not be used for real estate valuation. Because housing data can reflect historical and socioeconomic biases (e.g., neighborhood effects), predictions should be interpreted with care rather than treated as objective ground truth.

---

## Future Improvements

### Completed Improvements
- [x] Use Scikit-learn `Pipeline` and `ColumnTransformer` (implemented in `model_building.ipynb`)
- [x] Prevent data leakage by fitting preprocessing only on training data (done via pipeline)

### Planned Enhancements
- [ ] Tune polynomial degree dynamically
- [ ] Add SHAP explainability to web app
- [ ] Add more input fields to Streamlit app
- [ ] Deploy app to Streamlit Cloud
- [ ] Compare with LightGBM and CatBoost
- [ ] Add automated unit tests
- [ ] Implement real-time market adjustment factors
- [ ] Improve handling of unseen categorical values

---

## Author

```text
Priyal Aggarwal
```
---

## Contact

**Have questions or suggestions?**

- Email: priyal31706@gmail.com
- LinkedIn: www.linkedin.com/in/priyalaggarwal06 
- GitHub: [@Priyal9497](https://github.com/Priyal9497)

**Found this helpful?**  Star this repository!

## License

This project is licensed under the MIT License.