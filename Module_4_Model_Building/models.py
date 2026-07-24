from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor


def get_base_models() -> dict:
    return {
        'Polynomial Regression': LinearRegression(),   # fit on X_poly, not X_scaled
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.001, max_iter=10000),
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42, objective='reg:squarederror'),
    }


def train_models(models: dict, X_train, y_train) -> dict:
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models


def tune_ridge(X_train, y_train) -> GridSearchCV:
    grid = GridSearchCV(
        Ridge(), {'alpha': [0.01, 0.1, 1, 10, 50, 100]},
        scoring='neg_root_mean_squared_error', cv=5
    )
    grid.fit(X_train, y_train)
    return grid


def tune_lasso(X_train, y_train) -> GridSearchCV:
    grid = GridSearchCV(
        Lasso(max_iter=10000), {'alpha': [0.0001, 0.001, 0.01, 0.1, 1]},
        scoring='neg_root_mean_squared_error', cv=5
    )
    grid.fit(X_train, y_train)
    return grid


def tune_random_forest(X_train, y_train) -> RandomizedSearchCV:
    rf_params = {
        'n_estimators': [200, 400, 600],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2'],
    }
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1), rf_params,
        n_iter=20, scoring='neg_root_mean_squared_error', cv=5, random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search


def tune_xgboost(X_train, y_train) -> RandomizedSearchCV:
    xgb_params = {
        'n_estimators': [200, 400, 600],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.9, 1.0],
    }
    search = RandomizedSearchCV(
        XGBRegressor(random_state=42, objective='reg:squarederror'),
        xgb_params, n_iter=20,
        scoring='neg_root_mean_squared_error', cv=5, random_state=42
    )
    search.fit(X_train, y_train)
    return search


def tune_all_models(X_train, y_train) -> dict:
    ridge_grid = tune_ridge(X_train, y_train)
    lasso_grid = tune_lasso(X_train, y_train)
    rf_search = tune_random_forest(X_train, y_train)
    xgb_search = tune_xgboost(X_train, y_train)

    print("Best Ridge alpha:", ridge_grid.best_params_)
    print("Best Lasso alpha:", lasso_grid.best_params_)
    print("Best RF params:", rf_search.best_params_)
    print("Best XGBoost params:", xgb_search.best_params_)

    return {
        'Ridge Regression': ridge_grid.best_estimator_,
        'Lasso Regression': lasso_grid.best_estimator_,
        'Random Forest': rf_search.best_estimator_,
        'XGBoost': xgb_search.best_estimator_,
    }
