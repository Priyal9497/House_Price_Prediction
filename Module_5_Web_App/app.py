import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import plotly.graph_objects as go
import base64
import html
import time


# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
ASSETS_DIR = BASE_DIR / "assets"


# ============================================================
# Helper Functions
# ============================================================

def format_price(value):
    return f"${float(value):,.0f}"


def format_change(value):
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(float(value)):,.0f}"


def image_to_base64(path):
    path = Path(path)

    if not path.exists():
        return None

    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"

    with open(path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()

    return f"data:{mime};base64,{encoded}"


def get_background_image():
    local_bg = (
        image_to_base64(ASSETS_DIR / "house_bg.jpg")
        or image_to_base64(ASSETS_DIR / "house_bg.png")
    )

    if local_bg:
        return local_bg

    return (
        "https://images.unsplash.com/photo-1600566753190-17f0baa2a6c3"
        "?auto=format&fit=crop&w=1800&q=80"
    )


def inject_custom_css():
    bg_image = get_background_image()

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background:
                linear-gradient(135deg, rgba(2, 6, 23, 0.94), rgba(15, 23, 42, 0.78)),
                url("{bg_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}

        .block-container {{
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(2,6,23,0.98), rgba(15,23,42,0.94));
            border-right: 1px solid rgba(255,255,255,0.12);
        }}

        [data-testid="stSidebar"] * {{
            color: #f8fafc !important;
        }}

        .hero-card {{
            padding: 2.2rem 2.5rem;
            border-radius: 30px;
            background: linear-gradient(135deg, rgba(255,255,255,0.17), rgba(255,255,255,0.07));
            border: 1px solid rgba(255,255,255,0.22);
            box-shadow: 0 30px 80px rgba(0,0,0,0.38);
            backdrop-filter: blur(18px);
            margin-bottom: 1.5rem;
        }}

        .eyebrow {{
            color: #93c5fd;
            font-size: 0.82rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-weight: 800;
            margin-bottom: 0.6rem;
        }}

        .hero-title {{
            color: #ffffff;
            font-size: clamp(2.4rem, 5vw, 4.2rem);
            line-height: 1.05;
            margin: 0;
            font-weight: 800;
        }}

        .hero-subtitle {{
            color: #dbeafe;
            font-size: 1.05rem;
            max-width: 850px;
            line-height: 1.7;
            margin-top: 1rem;
        }}

        .model-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.55rem 0.95rem;
            border-radius: 999px;
            background: rgba(34, 197, 94, 0.16);
            color: #bbf7d0;
            border: 1px solid rgba(34, 197, 94, 0.38);
            font-size: 0.85rem;
            font-weight: 800;
            margin-top: 0.8rem;
        }}

        .glass-card {{
            padding: 1.3rem 1.5rem;
            border-radius: 24px;
            background: rgba(15, 23, 42, 0.62);
            border: 1px solid rgba(255,255,255,0.16);
            box-shadow: 0 18px 50px rgba(0,0,0,0.25);
            backdrop-filter: blur(16px);
            margin-bottom: 1rem;
        }}

        .section-title {{
            color: #ffffff;
            font-size: 1.25rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }}

        .soft-text {{
            color: #cbd5e1;
            font-size: 0.95rem;
            line-height: 1.65;
        }}

        .price-card {{
            padding: 1.8rem 2rem;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(249,115,22,0.95), rgba(239,68,68,0.95));
            border: 1px solid rgba(255,255,255,0.25);
            box-shadow: 0 25px 60px rgba(249,115,22,0.35);
            margin-bottom: 1.2rem;
        }}

        .price-label {{
            color: rgba(255,255,255,0.85);
            font-size: 0.95rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .price-value {{
            color: white;
            font-size: clamp(2.4rem, 5vw, 4.4rem);
            line-height: 1.05;
            font-weight: 800;
            margin-top: 0.35rem;
        }}

        .price-subtitle {{
            color: rgba(255,255,255,0.88);
            font-size: 1rem;
            margin-top: 0.7rem;
        }}

        [data-testid="stForm"] {{
            background: rgba(15, 23, 42, 0.64);
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 28px;
            padding: 1.4rem;
            box-shadow: 0 18px 50px rgba(0,0,0,0.25);
            backdrop-filter: blur(16px);
        }}

        label,
        [data-testid="stMarkdownContainer"] p,
        .stTabs [data-baseweb="tab"] p {{
            color: #f8fafc !important;
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: #ffffff !important;
        }}

        [data-testid="stNumberInput"] input {{
            background: rgba(255,255,255,0.93) !important;
            color: #0f172a !important;
            border-radius: 12px !important;
        }}

        [data-baseweb="select"] > div {{
            background: rgba(255,255,255,0.93) !important;
            border-radius: 12px !important;
        }}

        [data-baseweb="select"] span,
        [data-baseweb="select"] input {{
            color: #0f172a !important;
        }}

        div.stButton > button,
        div.stFormSubmitButton > button {{
            width: 100%;
            border: none;
            border-radius: 16px;
            padding: 0.85rem 1rem;
            font-weight: 800;
            color: white;
            background: linear-gradient(135deg, #ff4b4b, #f97316, #f59e0b);
            box-shadow: 0 16px 35px rgba(249,115,22,0.32);
            transition: all 0.2s ease-in-out;
        }}

        div.stButton > button:hover,
        div.stFormSubmitButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 22px 48px rgba(249,115,22,0.42);
            color: white;
        }}

        [data-testid="stMetric"] {{
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 18px;
            padding: 1rem;
        }}

        [data-testid="stMetric"] label,
        [data-testid="stMetric"] div {{
            color: #ffffff !important;
        }}

        hr {{
            border-color: rgba(255,255,255,0.12);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


inject_custom_css()


# ============================================================
# Load Artifacts
# ============================================================

@st.cache_resource
def load_artifacts():
    with open(ARTIFACT_DIR / "best_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open(ARTIFACT_DIR / "scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    with open(ARTIFACT_DIR / "feature_columns.pkl", "rb") as file:
        feature_columns = pickle.load(file)

    with open(ARTIFACT_DIR / "default_values.pkl", "rb") as file:
        default_values = pickle.load(file)

    with open(ARTIFACT_DIR / "best_model_name.pkl", "rb") as file:
        best_model_name = pickle.load(file)

    return model, scaler, feature_columns, default_values, best_model_name


try:
    model, scaler, feature_columns, default_values, best_model_name = load_artifacts()
except FileNotFoundError as error:
    st.error("Model artifacts were not found.")
    st.info("Please run `Module_4_Model_Building/model_building.ipynb` first to generate the required `.pkl` files.")
    st.exception(error)
    st.stop()


# Normalize artifact formats safely
feature_columns = list(feature_columns)

if isinstance(default_values, pd.Series):
    default_values = default_values.to_dict()
elif isinstance(default_values, pd.DataFrame):
    default_values = default_values.iloc[0].to_dict()

if isinstance(best_model_name, bytes):
    best_model_name = best_model_name.decode()

if isinstance(best_model_name, (list, tuple, np.ndarray)):
    best_model_name = best_model_name[0]

best_model_name = str(best_model_name)
safe_model_name = html.escape(best_model_name)


# ============================================================
# Optional Model Metrics
# ============================================================

def load_model_metrics():
    results_path = ARTIFACT_DIR / "model_results.csv"

    if not results_path.exists():
        return None, None

    try:
        results = pd.read_csv(results_path)

        def normalize_col(col):
            return (
                str(col).lower()
                .replace(" ", "")
                .replace("_", "")
                .replace("-", "")
                .replace("²", "2")
                .replace("^", "")
            )

        columns = {normalize_col(col): col for col in results.columns}

        model_col = columns.get("model") or columns.get("algorithm")
        rmse_col = columns.get("rmse")
        r2_col = columns.get("r2") or columns.get("r2score")

        selected_row = None

        if model_col:
            match = results[
                results[model_col].astype(str).str.lower().str.strip()
                == best_model_name.lower().strip()
            ]

            if not match.empty:
                selected_row = match.iloc[0]

        if selected_row is None and rmse_col:
            rmse_values = pd.to_numeric(results[rmse_col], errors="coerce")
            if rmse_values.notna().any():
                selected_row = results.loc[rmse_values.idxmin()]

        if selected_row is None:
            return None, None

        rmse = None
        r2 = None

        if rmse_col:
            rmse = pd.to_numeric(pd.Series([selected_row[rmse_col]]), errors="coerce").iloc[0]
            rmse = float(rmse) if pd.notna(rmse) else None

        if r2_col:
            r2 = pd.to_numeric(pd.Series([selected_row[r2_col]]), errors="coerce").iloc[0]
            r2 = float(r2) if pd.notna(r2) else None

        return rmse, r2

    except Exception:
        return None, None


model_rmse, model_r2 = load_model_metrics()


# ============================================================
# Neighborhood Options
# ============================================================

fallback_neighborhoods = [
    "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "Gilbert"
]

neighborhood_options = [
    col.replace("Neighborhood_", "")
    for col in feature_columns
    if str(col).startswith("Neighborhood_")
]

if not neighborhood_options:
    neighborhood_options = fallback_neighborhoods

neighborhood_options = sorted(set(neighborhood_options))

default_neighborhood = "NAmes" if "NAmes" in neighborhood_options else neighborhood_options[0]


# ============================================================
# Prediction Logic
# IMPORTANT: This keeps your original logic.
# ============================================================

def predict_house_price(values):
    overall_qual = values["overall_qual"]
    gr_liv_area = values["gr_liv_area"]
    garage_cars = values["garage_cars"]
    total_sf = values["total_sf"]
    year_built = values["year_built"]
    sale_year = values["sale_year"]
    full_bath = values["full_bath"]
    neighborhood = values["neighborhood"]

    # Start from realistic raw defaults
    input_dict = default_values.copy()

    # Update user-provided values
    input_dict["OverallQual"] = overall_qual
    input_dict["GrLivArea"] = gr_liv_area
    input_dict["GarageCars"] = garage_cars
    input_dict["TotalSF"] = total_sf
    input_dict["YearBuilt"] = year_built
    input_dict["YrSold"] = sale_year
    input_dict["HouseAge"] = sale_year - year_built
    input_dict["FullBath"] = full_bath

    # Update TotalBath if related columns exist
    half_bath = input_dict.get("HalfBath", 0)
    bsmt_full_bath = input_dict.get("BsmtFullBath", 0)
    bsmt_half_bath = input_dict.get("BsmtHalfBath", 0)

    input_dict["TotalBath"] = (
        full_bath
        + 0.5 * half_bath
        + bsmt_full_bath
        + 0.5 * bsmt_half_bath
    )

    # Reset neighborhood one-hot columns
    for col in feature_columns:
        if str(col).startswith("Neighborhood_"):
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

    return float(pred_price), input_df


# ============================================================
# Chart Functions
# ============================================================

def get_price_range(price):
    if model_rmse is not None and model_rmse > 1000:
        margin = model_rmse
        note = f"Using validation RMSE: {format_price(model_rmse)}"
    else:
        margin = price * 0.10
        note = "Illustrative ±10% range because final RMSE is unavailable."

    lower = max(0, price - margin)
    upper = price + margin

    return lower, upper, note


def create_price_gauge(price):
    axis_max = max(350000, price * 1.8)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=price,
            number={
                "prefix": "$",
                "valueformat": ",.0f",
                "font": {"size": 36, "color": "#ffffff"}
            },
            title={
                "text": "Predicted Sale Price",
                "font": {"size": 18, "color": "#e2e8f0"}
            },
            gauge={
                "axis": {
                    "range": [0, axis_max],
                    "tickprefix": "$",
                    "tickformat": ",.0f",
                    "tickcolor": "#e2e8f0"
                },
                "bar": {"color": "#f97316"},
                "bgcolor": "rgba(255,255,255,0.08)",
                "borderwidth": 1,
                "bordercolor": "rgba(255,255,255,0.18)",
                "steps": [
                    {
                        "range": [0, axis_max * 0.35],
                        "color": "rgba(59,130,246,0.25)"
                    },
                    {
                        "range": [axis_max * 0.35, axis_max * 0.70],
                        "color": "rgba(34,197,94,0.22)"
                    },
                    {
                        "range": [axis_max * 0.70, axis_max],
                        "color": "rgba(249,115,22,0.25)"
                    }
                ],
                "threshold": {
                    "line": {"color": "#22c55e", "width": 4},
                    "thickness": 0.75,
                    "value": price
                }
            }
        )
    )

    fig.update_layout(
        height=360,
        margin=dict(l=30, r=30, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff")
    )

    return fig


def create_price_range_chart(price, lower, upper, note):
    labels = ["Lower Estimate", "Predicted Price", "Upper Estimate"]
    values = [lower, price, upper]
    colors = ["#60a5fa", "#22c55e", "#f97316"]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            text=[format_price(v) for v in values],
            textposition="outside",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.35)", width=1)
            )
        )
    )

    fig.update_layout(
        title=f"Estimated Price Range<br><sup>{note}</sup>",
        height=360,
        yaxis=dict(
            title="Sale Price",
            tickprefix="$",
            tickformat=",.0f",
            gridcolor="rgba(255,255,255,0.12)"
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,0)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.04)",
        font=dict(color="#ffffff"),
        margin=dict(l=40, r=20, t=80, b=40)
    )

    return fig


def create_feature_profile_chart(values):
    house_age = values["sale_year"] - values["year_built"]

    categories = [
        "Quality",
        "Living Area",
        "Total SF",
        "Garage",
        "Bathrooms",
        "Newness"
    ]

    scores = [
        values["overall_qual"] / 10,
        min(values["gr_liv_area"] / 3500, 1),
        min(values["total_sf"] / 5500, 1),
        min(values["garage_cars"] / 4, 1),
        min(values["full_bath"] / 4, 1),
        max(0, min(1, 1 - house_age / 120))
    ]

    scores = [round(score * 100, 1) for score in scores]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=scores + [scores[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Feature Strength",
            line=dict(color="#38bdf8", width=3),
            fillcolor="rgba(56,189,248,0.25)"
        )
    )

    fig.update_layout(
        title="House Feature Profile",
        height=420,
        polar=dict(
            bgcolor="rgba(255,255,255,0.03)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color="#cbd5e1"),
                gridcolor="rgba(255,255,255,0.16)"
            ),
            angularaxis=dict(
                tickfont=dict(color="#ffffff", size=12),
                gridcolor="rgba(255,255,255,0.12)"
            )
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff"),
        margin=dict(l=40, r=40, t=70, b=40)
    )

    return fig


def create_workflow_chart():
    steps = [
        "User Inputs",
        "Default Values",
        "Feature Engineering",
        "One-Hot Columns",
        "Scaling",
        best_model_name,
        "Final Price"
    ]

    x = list(range(len(steps)))
    y = [1] * len(steps)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers+text",
            text=steps,
            textposition="bottom center",
            line=dict(color="#38bdf8", width=4),
            marker=dict(
                size=18,
                color="#f97316",
                line=dict(color="#ffffff", width=2)
            )
        )
    )

    fig.update_layout(
        title="How the Prediction is Generated",
        height=280,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#ffffff"),
        margin=dict(l=30, r=30, t=70, b=60)
    )

    return fig


def build_scenario_data(values, current_price):
    scenarios = [
        ("Current", {}),
        ("Quality +1", {"overall_qual": min(10, values["overall_qual"] + 1)}),
        (
            "Living Area +250",
            {
                "gr_liv_area": values["gr_liv_area"] + 250,
                "total_sf": values["total_sf"] + 250
            }
        ),
        ("Total SF +300", {"total_sf": values["total_sf"] + 300}),
        ("Garage +1", {"garage_cars": min(4, values["garage_cars"] + 1)}),
        ("Bathroom +1", {"full_bath": min(4, values["full_bath"] + 1)}),
        ("10 Years Newer", {"year_built": min(2026, values["year_built"] + 10)})
    ]

    if "NridgHt" in neighborhood_options and values["neighborhood"] != "NridgHt":
        scenarios.append(("Premium Area", {"neighborhood": "NridgHt"}))

    rows = []

    for scenario_name, changes in scenarios:
        scenario_values = values.copy()
        scenario_values.update(changes)

        try:
            if scenario_name == "Current":
                scenario_price = current_price
            else:
                scenario_price, _ = predict_house_price(scenario_values)

            rows.append(
                {
                    "Scenario": scenario_name,
                    "Estimated Price": scenario_price,
                    "Change vs Current": scenario_price - current_price
                }
            )

        except Exception:
            pass

    return pd.DataFrame(rows)


def create_scenario_chart(scenario_df):
    colors = [
        "#22c55e" if scenario == "Current" else "#38bdf8"
        for scenario in scenario_df["Scenario"]
    ]

    fig = go.Figure(
        go.Bar(
            x=scenario_df["Scenario"],
            y=scenario_df["Estimated Price"],
            text=[format_price(v) for v in scenario_df["Estimated Price"]],
            textposition="outside",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.35)", width=1)
            )
        )
    )

    fig.update_layout(
        title="What-if Price Comparison",
        height=430,
        yaxis=dict(
            title="Estimated Sale Price",
            tickprefix="$",
            tickformat=",.0f",
            gridcolor="rgba(255,255,255,0.12)"
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,0)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.04)",
        font=dict(color="#ffffff"),
        margin=dict(l=40, r=20, t=70, b=80)
    )

    return fig


# ============================================================
# Session State Defaults
# ============================================================

initial_values = {
    "overall_qual": 5,
    "gr_liv_area": 1500,
    "garage_cars": 2,
    "total_sf": 2000,
    "year_built": 2000,
    "sale_year": 2010,
    "full_bath": 2,
    "neighborhood": default_neighborhood
}

for key, value in initial_values.items():
    if key not in st.session_state:
        st.session_state[key] = value


def apply_preset(preset):
    for key, value in preset.items():
        if key == "neighborhood" and value not in neighborhood_options:
            value = default_neighborhood

        st.session_state[key] = value


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## 🧠 Model Studio")

    st.markdown(
        f"""
        <div class="glass-card">
            <div class="section-title">Active Model</div>
            <div class="model-pill">⚡ {safe_model_name}</div>
            <p class="soft-text">
                Trained on the Kaggle House Price dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.metric("Algorithm", best_model_name)

    if model_rmse is not None:
        st.metric("Validation RMSE", format_price(model_rmse))
    else:
        st.caption("Validation RMSE not available yet.")

    if model_r2 is not None:
        st.metric("R² Score", f"{model_r2:.3f}")

    st.divider()

    st.markdown("### ⚙️ Quick Presets")

    if st.button("🏘️ Starter Home"):
        apply_preset(
            {
                "overall_qual": 5,
                "gr_liv_area": 1200,
                "garage_cars": 1,
                "total_sf": 1700,
                "year_built": 1975,
                "sale_year": 2010,
                "full_bath": 1,
                "neighborhood": "NAmes"
            }
        )

    if st.button("🏡 Modern Family Home"):
        apply_preset(
            {
                "overall_qual": 7,
                "gr_liv_area": 1800,
                "garage_cars": 2,
                "total_sf": 2600,
                "year_built": 2005,
                "sale_year": 2010,
                "full_bath": 2,
                "neighborhood": "CollgCr"
            }
        )

    if st.button("🏰 Premium Home"):
        apply_preset(
            {
                "overall_qual": 9,
                "gr_liv_area": 2800,
                "garage_cars": 3,
                "total_sf": 3900,
                "year_built": 2008,
                "sale_year": 2010,
                "full_bath": 3,
                "neighborhood": "NridgHt"
            }
        )

    st.divider()

    st.caption(
        "Only key features are shown. Other required model features are filled using saved training defaults."
    )


# ============================================================
# Main Page
# ============================================================

st.markdown(
    f"""
    <div class="hero-card">
        <div class="eyebrow">AI-Powered Real Estate Valuation</div>
        <h1 class="hero-title">House Price Predictor</h1>
        <p class="hero-subtitle">
            Estimate a property's sale price using your trained Kaggle House Price regression model.
            Enter key home features, run the prediction, and explore visual insights through
            interactive graphs.
        </p>
        <div class="model-pill">🤖 Running Model: {safe_model_name}</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="glass-card">
        <div class="section-title">🏠 Enter Property Details</div>
        <div class="soft-text">
            Provide the main house details below. Features not entered by the user are automatically
            filled using the training dataset's median or mode values.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Input Form
# ============================================================

with st.form("prediction_form"):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.slider(
            "Overall Quality (1-10)",
            min_value=1,
            max_value=10,
            key="overall_qual"
        )

        st.number_input(
            "Above Ground Living Area (sq ft)",
            min_value=300,
            max_value=6000,
            step=50,
            key="gr_liv_area"
        )

        st.slider(
            "Garage Capacity (cars)",
            min_value=0,
            max_value=4,
            key="garage_cars"
        )

        st.number_input(
            "Total Square Footage",
            min_value=300,
            max_value=10000,
            step=50,
            key="total_sf"
        )

    with col2:
        st.number_input(
            "Year Built",
            min_value=1870,
            max_value=2026,
            step=1,
            key="year_built"
        )

        st.number_input(
            "Sale Year",
            min_value=2006,
            max_value=2026,
            step=1,
            key="sale_year"
        )

        st.slider(
            "Full Bathrooms",
            min_value=0,
            max_value=4,
            key="full_bath"
        )

        st.selectbox(
            "Neighborhood",
            options=neighborhood_options,
            key="neighborhood"
        )

    submitted = st.form_submit_button("✨ Predict Price")


def collect_input_values():
    return {
        "overall_qual": int(st.session_state["overall_qual"]),
        "gr_liv_area": int(st.session_state["gr_liv_area"]),
        "garage_cars": int(st.session_state["garage_cars"]),
        "total_sf": int(st.session_state["total_sf"]),
        "year_built": int(st.session_state["year_built"]),
        "sale_year": int(st.session_state["sale_year"]),
        "full_bath": int(st.session_state["full_bath"]),
        "neighborhood": str(st.session_state["neighborhood"])
    }


def run_prediction_animation(values):
    if hasattr(st, "chat_message"):
        container = st.chat_message("assistant", avatar="🤖")
    else:
        container = st.container()

    with container:
        if hasattr(st, "status"):
            with st.status("Thinking through the house valuation...", expanded=True) as status:
                st.write("🔍 Reading user-provided house features...")
                time.sleep(0.35)

                st.write("🧩 Filling missing fields using training-set default values...")
                time.sleep(0.35)

                st.write(
                    f"🏗️ Creating engineered features: "
                    f"HouseAge = {values['sale_year'] - values['year_built']} years, "
                    f"TotalBath calculated from bathroom fields..."
                )
                time.sleep(0.35)

                st.write("🏘️ Encoding neighborhood information...")
                time.sleep(0.35)

                st.write("📐 Aligning input columns with the exact training feature order...")
                time.sleep(0.35)

                st.write("⚙️ Applying the saved scaler...")
                time.sleep(0.35)

                st.write(f"🚀 Running the **{best_model_name}** model...")
                pred_price, input_df = predict_house_price(values)
                time.sleep(0.35)

                st.write("💰 Converting log prediction back to dollar price using `np.expm1()`...")
                time.sleep(0.25)

                status.update(
                    label=f"Prediction complete — {best_model_name} model used",
                    state="complete",
                    expanded=False
                )

                return pred_price, input_df

        with st.spinner(f"Running {best_model_name} model..."):
            pred_price, input_df = predict_house_price(values)
            return pred_price, input_df


# ============================================================
# Prediction
# ============================================================

if submitted:
    values = collect_input_values()

    if values["sale_year"] < values["year_built"]:
        st.warning(
            "Sale Year is earlier than Year Built. The prediction will still run, "
            "but please check if the input is correct."
        )

    if values["total_sf"] < values["gr_liv_area"]:
        st.warning(
            "Total Square Footage is usually greater than or equal to Above Ground Living Area. "
            "The prediction will still run, but please verify the input."
        )

    try:
        pred_price, input_df = run_prediction_animation(values)

        scenario_df = build_scenario_data(values, pred_price)

        st.session_state["last_prediction"] = {
            "price": pred_price,
            "values": values,
            "input_df": input_df,
            "scenario_df": scenario_df
        }

    except Exception as error:
        st.error("Prediction failed. This may be caused by a mismatch between the saved model artifacts and app input features.")
        st.exception(error)


# ============================================================
# Results Dashboard
# ============================================================

if "last_prediction" in st.session_state:
    result = st.session_state["last_prediction"]

    pred_price = result["price"]
    values = result["values"]
    input_df = result["input_df"]
    scenario_df = result["scenario_df"]

    lower_price, upper_price, range_note = get_price_range(pred_price)
    house_age = values["sale_year"] - values["year_built"]

    st.markdown("---")

    st.markdown(
        f"""
        <div class="price-card">
            <div class="price-label">Estimated Sale Price</div>
            <div class="price-value">{format_price(pred_price)}</div>
            <div class="price-subtitle">
                Generated using the {safe_model_name} model trained on Kaggle House Price data.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    metric_col1.metric("Predicted Price", format_price(pred_price))
    metric_col2.metric("Estimated Range", f"{format_price(lower_price)} - {format_price(upper_price)}")
    metric_col3.metric("Model Used", best_model_name)
    metric_col4.metric("House Age", f"{house_age} years")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📈 Price Graphs",
            "🔁 What-if Analysis",
            "🧠 Model Thinking",
            "📋 Input Used"
        ]
    )

    with tab1:
        graph_col1, graph_col2 = st.columns(2)

        with graph_col1:
            st.plotly_chart(
                create_price_gauge(pred_price),
                use_container_width=True
            )

        with graph_col2:
            st.plotly_chart(
                create_price_range_chart(pred_price, lower_price, upper_price, range_note),
                use_container_width=True
            )

        st.plotly_chart(
            create_feature_profile_chart(values),
            use_container_width=True
        )

    with tab2:
        st.plotly_chart(
            create_scenario_chart(scenario_df),
            use_container_width=True
        )

        display_df = scenario_df.copy()
        display_df["Estimated Price"] = display_df["Estimated Price"].apply(format_price)
        display_df["Change vs Current"] = display_df["Change vs Current"].apply(format_change)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

    with tab3:
        st.plotly_chart(
            create_workflow_chart(),
            use_container_width=True
        )

        st.markdown(
            f"""
            <div class="glass-card">
                <div class="section-title">🤖 How the Result Was Created</div>
                <div class="soft-text">
                    <b>Model working behind the app:</b> {safe_model_name}
                    <br><br>
                    The prediction is generated using the same machine learning pipeline created during model training:
                    <ol>
                        <li>User inputs are collected from the web form.</li>
                        <li>Missing fields are filled using training-set default values.</li>
                        <li>Engineered features such as <b>HouseAge</b> and <b>TotalBath</b> are calculated.</li>
                        <li>Neighborhood columns are encoded using one-hot encoded feature columns.</li>
                        <li>The final input is reordered to match <b>feature_columns.pkl</b>.</li>
                        <li>The saved scaler transforms the input.</li>
                        <li>The <b>{safe_model_name}</b> model predicts the log sale price.</li>
                        <li>The output is converted back to dollars using <b>np.expm1()</b>.</li>
                    </ol>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab4:
        user_input_summary = pd.DataFrame(
            {
                "Feature": [
                    "Overall Quality",
                    "Above Ground Living Area",
                    "Garage Capacity",
                    "Total Square Footage",
                    "Year Built",
                    "Sale Year",
                    "House Age",
                    "Full Bathrooms",
                    "Neighborhood"
                ],
                "Value": [
                    values["overall_qual"],
                    f"{values['gr_liv_area']:,} sq ft",
                    values["garage_cars"],
                    f"{values['total_sf']:,} sq ft",
                    values["year_built"],
                    values["sale_year"],
                    f"{house_age} years",
                    values["full_bath"],
                    values["neighborhood"]
                ]
            }
        )

        st.dataframe(
            user_input_summary,
            use_container_width=True,
            hide_index=True
        )

        with st.expander("View complete model input after preprocessing"):
            processed_input = input_df.T.reset_index()
            processed_input.columns = ["Feature", "Value"]

            st.dataframe(
                processed_input,
                use_container_width=True,
                height=400
            )


# ============================================================
# Footer
# ============================================================

st.markdown(
    """
    <br>
    <div class="glass-card">
        <div class="soft-text">
            ⚠️ This prediction is for educational purposes only. Actual real estate prices may be affected by inflation,
            interest rates, local demand, renovations, school districts, and current market conditions not included in
            the Kaggle House Price dataset.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)