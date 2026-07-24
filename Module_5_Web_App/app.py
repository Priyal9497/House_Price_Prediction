import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

st.set_page_config(page_title="House Price Predictor", layout="centered")

# Load artifacts
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

model = pickle.load(open(ARTIFACT_DIR / "best_model.pkl", "rb"))
scaler = pickle.load(open(ARTIFACT_DIR / "scaler.pkl", "rb"))
feature_columns = pickle.load(open(ARTIFACT_DIR / "feature_columns.pkl", "rb"))
default_values = pickle.load(open(ARTIFACT_DIR / "default_values.pkl", "rb"))
best_model_name = pickle.load(open(ARTIFACT_DIR / "best_model_name.pkl", "rb"))

st.title("House Price Predictor")

st.write(
    "Enter key house features below. Missing features are filled using typical "
    "training-set values such as medians or modes."
)

with st.sidebar:
    st.header("Model Info")
    st.write(f"**Algorithm:** {best_model_name}")
    st.caption("Trained on the Kaggle Ames Housing dataset.")

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", 300, 6000, 1500)
    garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 2)
    total_sf = st.number_input("Total Square Footage", 300, 10000, 2000)

with col2:
    year_built = st.number_input("Year Built", 1870, 2026, 2000)
    sale_year = st.number_input("Sale Year", 2006, 2026, 2010)
    full_bath = st.slider("Full Bathrooms", 0, 4, 2)
    neighborhood = st.selectbox(
        "Neighborhood",
        ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert']
    )

if st.button("Predict Price", type="primary"):

    # Start from realistic raw defaults
    input_dict = default_values.copy()

    # Update user-provided values
    input_dict['OverallQual'] = overall_qual
    input_dict['GrLivArea'] = gr_liv_area
    input_dict['GarageCars'] = garage_cars
    input_dict['TotalSF'] = total_sf
    input_dict['YearBuilt'] = year_built
    input_dict['YrSold'] = sale_year
    input_dict['HouseAge'] = sale_year - year_built
    input_dict['FullBath'] = full_bath

    # Update TotalBath if related columns exist
    half_bath = input_dict.get('HalfBath', 0)
    bsmt_full_bath = input_dict.get('BsmtFullBath', 0)
    bsmt_half_bath = input_dict.get('BsmtHalfBath', 0)

    input_dict['TotalBath'] = (
        full_bath
        + 0.5 * half_bath
        + bsmt_full_bath
        + 0.5 * bsmt_half_bath
    )

    # Reset neighborhood one-hot columns
    for col in feature_columns:
        if col.startswith("Neighborhood_"):
            input_dict[col] = 0

    neighborhood_col = f"Neighborhood_{neighborhood}"
    if neighborhood_col in input_dict:
        input_dict[neighborhood_col] = 1

    # Build input dataframe in exact training column order
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Scale raw input values
    input_scaled = scaler.transform(input_df)

    # Predict log price, then inverse transform
    pred_log = model.predict(input_scaled)[0]
    pred_price = np.expm1(pred_log)

    st.success(f"### Predicted Sale Price: ${pred_price:,.0f}")

    st.caption(
        "Note: This is a model-based estimate trained on historical Ames Housing data. "
        "It may not reflect current market inflation or local real-time conditions."
    )