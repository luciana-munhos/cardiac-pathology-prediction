from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Trains a Random Forest classifier on the given training data with optional 
    control over the number of estimators and random state.
    """
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    return rf

def train_model_with_cv(model, param_grid, X_train, y_train, X_test=None,
                        cv=5, n_jobs=-1, verbose=1, scoring='accuracy'):
    """
    Performs hyperparameter tuning using GridSearchCV and trains the given model.

    Parameters:
        model: Untrained model instance (e.g., RandomForestClassifier(), XGBClassifier(), etc.)
        param_grid (dict): Grid of hyperparameters to search
        X_train: Training feature matrix
        y_train: Training labels
        X_test (optional): Feature matrix for test data (if you want predictions)
        cv (int): Number of cross-validation folds
        n_jobs (int): Parallel jobs
        verbose (int): Verbosity level
        scoring (str): Metric for scoring grid search

    Returns:
        best_model: Best estimator from GridSearchCV
        y_pred: Predictions on X_test (if provided), else None
        best_params: Best hyperparameters found
    """
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"Best parameters: {best_params}")
    print(f"Best {scoring} (CV): {grid_search.best_score_:.4f}")

    y_pred = best_model.predict(X_test) if X_test is not None else None

    return best_model, y_pred, best_params
