import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def lasso_important_features(trained_lasso, X_train, threshold: float = 0.001) -> pd.Series:
    """Method 1 — Lasso coefficients (near-zero coefficients = unimportant)."""
    coefs = pd.Series(trained_lasso.coef_, index=X_train.columns)
    important = coefs[coefs.abs() > threshold].sort_values(key=abs, ascending=False)
    return important


def tree_feature_importance(trained_tree_model, X_train) -> pd.Series:
    """Method 2 — feature importances from a tree-based model (RF or XGBoost)."""
    return pd.Series(
        trained_tree_model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)


def rfe_selected_features(X_train, y_train, n_features: int = 20) -> pd.Index:
    """Method 3 — Recursive Feature Elimination (example on Ridge)."""
    rfe = RFE(estimator=Ridge(alpha=1.0), n_features_to_select=n_features)
    rfe.fit(X_train, y_train)
    return X_train.columns[rfe.support_]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_models(trained_models: dict, X_test, X_test_poly, y_test) -> pd.DataFrame:
    """
    Evaluate all trained models.

    Parameters
    ----------
    trained_models : dict
        Dictionary of trained models.
    X_test : DataFrame
        Scaled test features for all models except Polynomial Regression.
    X_test_poly : DataFrame
        Test features with polynomial terms for Polynomial Regression.
    y_test : Series
        Log-transformed target values.
    """

    results = []

    actual = np.expm1(y_test)

    for name, model in trained_models.items():

        # Polynomial Regression uses polynomial features
        if name == "Polynomial Regression":
            preds_log = model.predict(X_test_poly)
        else:
            preds_log = model.predict(X_test)

        preds = np.expm1(preds_log)

        mse = mean_squared_error(actual, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, preds)
        r2 = r2_score(actual, preds)

        results.append({
            "Model": name,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

    return (pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True))


def plot_rmse_comparison(results_df: pd.DataFrame, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.barh(results_df['Model'], results_df['RMSE'], color='steelblue')
    plt.xlabel('RMSE ($)')
    plt.title('Model Comparison — RMSE (lower is better)')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
