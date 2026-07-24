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

Files used:

```text
train.csv
test.csv
data_description.txt
```

The target variable is:

```text
SalePrice
```

The dataset contains many numerical and categorical features related to house characteristics, such as:

- Overall house quality
- Living area
- Lot size
- Neighborhood
- Garage size
- Basement area
- Number of bathrooms
- Year built
- Sale condition

Because `SalePrice` is right-skewed, a log transformation is applied during model training:

```python
SalePrice_Log = np.log1p(SalePrice)
```

Predicted values are converted back to the original price scale using:

```python
np.expm1(prediction)
```

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

## Modules

## Module 1: Research Report

This module contains the theoretical research report for the project.

Topics covered include:

- Data Analytics
- Data Science
- Artificial Intelligence
- Advanced Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regression
- XGBoost
- Feature selection
- Hyperparameter tuning
- Real estate market complexity
- Ethical considerations in house price prediction

Deliverable:

```text
Module_1_Research_Report/research_report.pdf
```

**Status:** Complete — See `research_report.pdf`

---

## Module 2: Action Plan

This module contains the project plan and timeline.

It includes:

- Project objective
- Dataset selection
- Environment setup
- EDA plan
- Data preprocessing plan
- Model building plan
- Evaluation plan
- Web app plan
- Timeline

Deliverable:

```text
Module_2_Action_Plan/action_plan.pdf
```

**Status:** Complete — See `action_plan.pdf`

---

## Module 3: Exploratory Data Analysis and Hypothesis Testing

This module analyzes the housing dataset to understand patterns, relationships, missing values, outliers, and important price-related features.

EDA tasks include:

- Loading and inspecting the dataset
- Checking data types
- Identifying missing values
- Analyzing the target variable `SalePrice`
- Applying log transformation to `SalePrice`
- Analyzing numerical features
- Analyzing categorical features
- Correlation analysis
- Outlier detection
- Neighborhood price comparison
- Visualizations using Matplotlib and Seaborn

Visualizations include:

- Histograms
- Q-Q plots
- Scatter plots
- Box plots
- Bar plots
- Correlation heatmaps

Deliverables:

```text
Module_3_EDA/eda_analysis.ipynb
Module_3_EDA/visualizations/
```

**Status:** Complete — includes a written key-insights summary and documented preprocessing decisions carried into Module 4.

---

## Hypothesis Testing

Three hypotheses were tested during EDA.

### Hypothesis 1: Living Area and House Price

```text
H0: There is no linear correlation between GrLivArea and SalePrice.
H1: There is a significant positive correlation between GrLivArea and SalePrice.
```

Test used:

```text
Pearson correlation test
```

Result: Rejected H0 — significant positive correlation (r ≈ 0.71, p < 0.001).

---

### Hypothesis 2: Neighborhood and House Price

```text
H0: Average SalePrice does not differ significantly across neighborhoods.
H1: At least one neighborhood has a significantly different average SalePrice.
```

Test used:

```text
One-way ANOVA
```

Result: Rejected H0 — average price differs significantly across neighborhoods (F ≈ 71.78, p < 0.001).

---

### Hypothesis 3: House Age and House Price

```text
H0: There is no linear correlation between HouseAge and SalePrice.
H1: HouseAge has a significant negative correlation with SalePrice.
```

Test used:

```text
Pearson correlation test
```

Result: Rejected H0 — significant negative correlation (r ≈ -0.52, p < 0.001).

---

## Module 4: Model Building, Prediction, and Evaluation

This module contains the complete machine learning workflow.

Main steps:

1. Load the dataset.
2. Remove extreme outliers identified during EDA.
3. Handle missing values.
4. Encode categorical variables.
5. Engineer new features.
6. Scale features.
7. Create polynomial features.
8. Split data into training and testing sets.
9. Train regression models.
10. Apply feature selection.
11. Tune hyperparameters.
12. Evaluate model performance.
13. Compare all models.
14. Save the best model for deployment.

Deliverables:

```text
Module_4_Model_Building/preprocessing.py
Module_4_Model_Building/models.py
Module_4_Model_Building/evaluation.py
Module_4_Model_Building/model_building.ipynb
```

**Status:** Complete — All Python modules implemented. Pending final run on full dataset to populate performance metrics.

---

## Feature Engineering

The following engineered features were created:

```text
HouseAge = YrSold - YearBuilt
RemodAge = YrSold - YearRemodAdd
TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
TotalBath = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath
```

These features help capture important housing characteristics such as total living space, age of the house, remodeling age, and total bathroom count.

---

## Machine Learning Models Used

The following regression models were implemented:

### 1. Polynomial Regression

Polynomial Regression was used to capture non-linear relationships between selected important predictors and house prices.

### 2. Ridge Regression

Ridge Regression applies L2 regularization to reduce overfitting and handle multicollinearity.

### 3. Lasso Regression

Lasso Regression applies L1 regularization and can shrink less important feature coefficients to zero, making it useful for feature selection.

### 4. Random Forest Regression

Random Forest is an ensemble model that can capture non-linear relationships and interactions between features.

### 5. XGBoost Regression

XGBoost is a gradient boosting algorithm that performs well on structured/tabular datasets and is commonly used in regression competitions.

---

## Feature Selection

Feature selection was performed using:

- Lasso coefficients
- Random Forest feature importance
- XGBoost feature importance
- Recursive Feature Elimination

Important features identified during analysis included:

- OverallQual
- GrLivArea
- TotalSF
- GarageCars
- GarageArea
- TotalBsmtSF
- 1stFlrSF
- YearBuilt
- HouseAge
- Neighborhood-related features

---

## Hyperparameter Tuning

Hyperparameter tuning was performed using:

```text
GridSearchCV
RandomizedSearchCV
```

Tuned models include:

- Ridge Regression
- Lasso Regression
- Random Forest Regression
- XGBoost Regression

The goal of hyperparameter tuning was to improve model performance and reduce prediction error.

---

## Evaluation Metrics

Models were evaluated using the following regression metrics:

| Metric | Meaning |
|---|---|
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| R² | Coefficient of Determination |

RMSE was used as the main comparison metric because it shows the average prediction error in house price dollar scale.

---

## Model Evaluation Results

After training and tuning, all models were compared using MSE, RMSE, MAE, and R².

The final results will be saved in:

```text
Module_5_Web_App/artifacts/model_results.csv
```

>**NOTE:** Run `Module_4_Model_Building/model_building.ipynb` to populate these values with actual results.

| Model | MSE | RMSE | MAE | R² |
|---|---:|---:|---:|---:|
| Polynomial Regression | *Pending* | *Pending* | *Pending* | *Pending* |
| Ridge Regression | *Pending*  | *Pending*  | *Pending*  | *Pending*  |
| Lasso Regression | *Pending*  | *Pending* | *Pending* | Add *Pending* |
| Random Forest | *Pending* | *Pending* | *Pending* | *Pending* |
| XGBoost | *Pending* | *Pending* | Add value | Add value |

The model with the lowest RMSE will be selected as the final model for deployment.

---

## Key Findings

The EDA and model analysis showed that:

1. `SalePrice` is right-skewed and benefits from log transformation.
2. `OverallQual` is one of the strongest predictors of house price.
3. `GrLivArea` has a strong positive relationship with `SalePrice`.
4. Neighborhood has a significant effect on house prices.
5. Newer houses generally have higher sale prices.
6. Garage capacity and garage area are important price predictors.
7. Basement size and total square footage strongly influence house price.
8. Some missing values represent the absence of a feature, such as no pool, no garage, no basement, or no alley.
9. Outliers can affect model training and were reviewed during EDA — two `GrLivArea` outliers (Id 524, Id 1299) are removed before model training.

---

## Module 5: Streamlit Web Application

A Streamlit web application will allow users to input house features and receive a predicted sale price.

The app is located at:

```text
Module_5_Web_App/app.py
```

The app uses saved model artifacts from:

```text
Module_5_Web_App/artifacts/
```

Required artifact files:

```text
best_model.pkl
scaler.pkl
feature_columns.pkl
default_values.pkl
best_model_name.pkl
model_results.csv
```

The app will allow users to enter features such as:

- Overall quality
- Above-ground living area
- Garage capacity
- Total square footage
- Year built
- Sale year
- Full bathrooms
- Neighborhood

Features not entered by the user are filled using default median/mode values from the training data.

**Status:** Code complete — Pending model artifacts from Module 4. Ready to deploy once `best_model.pkl` is generated.

---
---
## 🚀 Quick Start

**Want to try the app immediately?**

```bash
# 1. Clone and setup
git clone https://github.com/Priyal9497/House_Price_Prediction.git
cd House_Price_Prediction
pip install -r requirements.txt

# 2. Download dataset to data/raw/
# Get train.csv and test.csv from Kaggle

# 3. Run model training
jupyter notebook Module_4_Model_Building/model_building.ipynb
# Run all cells to generate model artifacts

# 4. Launch web app
cd Module_5_Web_App
streamlit run app.py'''
---


## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Priyal9497/House_Price_Prediction.git
cd House_Price_Prediction
```


---

### 2. Create a Virtual Environment

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

For macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run the Project

### Run EDA Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open and run:

```text
Module_3_EDA/eda_analysis.ipynb
```

---

### Run Model Building Notebook

Open and run:

```text
Module_4_Model_Building/model_building.ipynb
```

This notebook will:

- preprocess the data,
- train models,
- tune hyperparameters,
- evaluate performance,
- save model artifacts.

After running it successfully, the following files should be created:

```text
Module_5_Web_App/artifacts/best_model.pkl
Module_5_Web_App/artifacts/scaler.pkl
Module_5_Web_App/artifacts/feature_columns.pkl
Module_5_Web_App/artifacts/default_values.pkl
Module_5_Web_App/artifacts/best_model_name.pkl
Module_5_Web_App/artifacts/model_results.csv
```

---

### Run the Web App

From the project root folder:

```bash
cd Module_5_Web_App
streamlit run app.py
```

The app will open in your browser.

---

## Requirements

Main libraries used:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
scipy
streamlit
joblib
jupyter
ipykernel
plotly
statsmodels
```

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

## Important Notes

- The model is trained on historical Ames Housing data.
- Predictions may not reflect current real estate market prices.
- This project is for educational and academic purposes.
- The model should not be used as the only source for real estate valuation.
- External market factors such as inflation, interest rates, school districts, demand, and economic conditions are not included in the dataset.

---

## Ethical Considerations

House price prediction models must be used carefully because real estate data may contain historical or social biases.

Potential ethical issues include:

- Neighborhood-based bias
- Socioeconomic bias
- Discriminatory pricing effects
- Reinforcement of historical inequalities
- Overreliance on automated predictions

This project is intended for learning and demonstration purposes only.

---

## Troubleshooting

### Error: `best_model.pkl` not found

If you see:

```text
FileNotFoundError: best_model.pkl not found
```

Run all cells in:

```text
Module_4_Model_Building/model_building.ipynb
```

Make sure this folder exists:

```text
Module_5_Web_App/artifacts/
```

And contains:

```text
best_model.pkl
scaler.pkl
feature_columns.pkl
default_values.pkl
best_model_name.pkl
model_results.csv
```

---

### Error: Empty `.pkl` files

Do not manually create `.pkl` files.

They must be generated by the model building notebook.

Correct extension:

```text
.pkl
```

Incorrect extension:

```text
.pk1
```

The correct extension uses the letter `l`, not the number `1`.

---

### Error: Feature mismatch in Streamlit app

Make sure:

1. `feature_columns.pkl` was saved from the same training pipeline.
2. `default_values.pkl` was saved correctly.
3. The input DataFrame uses the same column order as training data.
4. The same scaler used during training is loaded in the app.

---

## Future Improvements

### ✅ Completed Improvements
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

## Project Status

```text

**SHOULD BE:**
```markdown
## Project Status

**Current Status:** **90% Complete**

| Module | Status | Notes |
|--------|--------|-------|
| Module 1: Research Report | Complete | PDF document finished |
| Module 2: Action Plan | Complete | PDF document finished |
| Module 3: EDA | Complete | Notebook with visualizations and hypothesis tests |
| Module 4: Model Building | In Progress | Code complete, pending final metrics from full dataset run |
| Module 5: Web App | Ready | Code complete, pending model artifacts |
| Module 6: GitHub Repo | In Progress | Documentation finalization |

**Next Steps:**
1. Run `model_building.ipynb` on full dataset to generate final metrics
2. Update README with final RMSE/R² values
3. Push to GitHub
4. Optional: Deploy Streamlit app to cloud
```

---

## Contact

**Have questions or suggestions?**

- Email: priyal31706@gmail.com
- LinkedIn: www.linkedin.com/in/priyalaggarwal06 
- GitHub: [@Priyal9497](https://github.com/Priyal9497)

**Found this helpful?** ⭐ Star this repository!

## License

## License

This project is licensed under the MIT License.

Note: The dataset used in this project comes from Kaggle's House Prices: Advanced Regression Techniques competition and is used for educational purposes.