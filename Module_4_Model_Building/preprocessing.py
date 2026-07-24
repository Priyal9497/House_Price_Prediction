import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Categorical columns where NaN genuinely means "does not have this feature"
NONE_COLS = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'MasVnrType'
]

# Numerical columns where NaN means "0 of this feature" (e.g. no garage -> 0 cars)
ZERO_COLS = [
    'GarageYrBlt', 'GarageArea', 'GarageCars',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
]

# Ordinal quality-scale mapping (rank order matters here)
QUAL_MAP = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
QUAL_COLS = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
    'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond'
]

# Strongest numeric predictors from EDA -> used for polynomial features
TOP_NUMERIC = ['OverallQual', 'GrLivArea', 'TotalSF', 'GarageCars']


def load_and_clean_target(csv_path: str) -> pd.DataFrame:
    train = pd.read_csv(csv_path)

    # Drop the GrLivArea outliers identified in EDA
    train = train[
        ~((train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000))
    ].reset_index(drop=True)

    # Log-transform the target (predict this, then np.expm1() to invert at the end)
    train['SalePrice_Log'] = np.log1p(train['SalePrice'])
    return train


def handle_missing_values(train: pd.DataFrame) -> pd.DataFrame:
    train = train.copy()

    for col in NONE_COLS:
        train[col] = train[col].fillna('None')

    for col in ZERO_COLS:
        train[col] = train[col].fillna(0)

    # LotFrontage: impute by neighborhood median (similar lot sizes per neighborhood)
    train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)

    train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].median())

    # Anything left (rare categoricals like Electrical) -> impute with mode
    for col in train.columns:
        if train[col].isnull().sum() > 0 and train[col].dtype == 'object':
            train[col] = train[col].fillna(train[col].mode()[0])

    return train


def encode_categoricals(train: pd.DataFrame) -> pd.DataFrame:
    train = train.copy()

    for col in QUAL_COLS:
        if train[col].isnull().sum() > 0:
            train[col] = train[col].fillna(0)

    train = pd.get_dummies(train, drop_first=True)
    return train


def engineer_features(train: pd.DataFrame) -> pd.DataFrame:
    train = train.copy()

    train['HouseAge'] = train['YrSold'] - train['YearBuilt']
    train['RemodAge'] = train['YrSold'] - train['YearRemodAdd']
    train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
    train['TotalBath'] = (
        train['FullBath'] + 0.5 * train['HalfBath']
        + train['BsmtFullBath'] + 0.5 * train['BsmtHalfBath']
    )
    return train


def scale_and_add_poly(train: pd.DataFrame):
    """
    Split into X/y, scale features, and build a polynomial-feature version
    (X_poly) for the Polynomial Regression model only.

    Returns:
        X_scaled : Original scaled features
        X_poly   : Original scaled features + new polynomial features
        y        : Log-transformed target
        scaler   : Fitted StandardScaler
    """
    # Separate features and target
    X = train.drop(columns=['Id', 'SalePrice', 'SalePrice_Log'])
    y = train['SalePrice_Log']

    # Scale all features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    # Generate polynomial features for only the top numeric predictors
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(X_scaled[TOP_NUMERIC])
    poly_cols = poly.get_feature_names_out(TOP_NUMERIC)
    poly_df = pd.DataFrame(
        poly_features,
        columns=poly_cols,
        index=X_scaled.index
    )
    # Remove duplicate original columns because they already exist in X_scaled
    poly_df = poly_df.drop(columns=TOP_NUMERIC)

    # Combine original scaled features with only the NEW polynomial features
    X_poly = pd.concat([X_scaled, poly_df], axis=1)

    return X_scaled, X_poly, y, scaler


def run_full_pipeline(csv_path: str):
    train = load_and_clean_target(csv_path)
    train = handle_missing_values(train)
    train = encode_categoricals(train)
    train = engineer_features(train)
    return scale_and_add_poly(train)
