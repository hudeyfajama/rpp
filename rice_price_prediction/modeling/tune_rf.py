"""
Hyperparameter tuning for Random Forest model.

This module uses GridSearchCV to find optimal hyperparameters
for the Random Forest regressor.
"""

from pathlib import Path
import joblib

import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import typer

from rice_price_prediction.config import MODELS_DIR, PROCESSED_DATA_DIR
from rice_price_prediction.features import preprocess_data

app = typer.Typer()


def get_rf_param_grid():
    """
    Get parameter grid for Random Forest tuning.

    Returns:
    --------
    dict
        Parameter grid for hyperparameter search
    """
    # Reduced grid for faster search
    # Total combinations: 2 * 2 * 2 * 2 * 1 = 16 (much faster!)
    param_grid = {
        'n_estimators': [100, 200],              # Reduced from 3 to 2
        'max_depth': [15, 20],                   # Removed None and 10
        'min_samples_split': [5, 10],            # Removed 2 (too granular)
        'min_samples_leaf': [2, 4],              # Removed 1 (causes overfitting)
        'max_features': ['sqrt'],                # Keep only best option
        'bootstrap': [True],                     # Keep as is
    }

    return param_grid


@app.command()
def tune(
    data_path: Path = PROCESSED_DATA_DIR / "rice_prices_cleaned.csv",
    output_path: Path = MODELS_DIR / "rf_tuned_model.pkl",
    cv: int = typer.Option(5, help="Number of cross-validation folds"),
    n_jobs: int = typer.Option(-1, help="Number of parallel jobs (-1 for all cores)"),
    test_size: float = 0.2,
):
    """
    Tune Random Forest hyperparameters using grid search.

    Parameters:
    -----------
    data_path : Path
        Path to cleaned dataset
    output_path : Path
        Path to save tuned model
    cv : int
        Number of cross-validation folds
    n_jobs : int
        Number of parallel jobs
    test_size : float
        Test set proportion
    """
    logger.info("="*60)
    logger.info("RANDOM FOREST HYPERPARAMETER TUNING - GRID SEARCH")
    logger.info("="*60)

    # Load and preprocess data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df):,} records")

    logger.info("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, label_encoders = preprocess_data(
        df, test_size=test_size
    )

    # Get parameter grid
    param_grid = get_rf_param_grid()

    logger.info(f"\n{'='*60}")
    logger.info("PARAMETER GRID")
    logger.info("="*60)
    for param, values in param_grid.items():
        logger.info(f"  {param}: {values}")

    # Create base model
    rf_base = RandomForestRegressor(random_state=42, n_jobs=1)  # n_jobs handled by CV

    # Setup grid search
    logger.info(f"\n{'='*60}")
    logger.info("STARTING GRID SEARCH")
    logger.info("="*60)
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    logger.info(f"Total parameter combinations: {total_combinations}")
    logger.info(f"Cross-validation folds: {cv}")
    logger.info(f"Total fits: {total_combinations * cv}")
    logger.warning("This may take a long time...")

    search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=n_jobs,
        verbose=2,
        return_train_score=True
    )

    # Fit search
    logger.info("\nStarting hyperparameter search...")
    search.fit(X_train, y_train)

    # Results
    logger.info(f"\n{'='*60}")
    logger.info("SEARCH RESULTS")
    logger.info("="*60)

    logger.success(f"Best score (CV MSE): {-search.best_score_:.4f}")
    logger.info("\nBest parameters:")
    for param, value in search.best_params_.items():
        logger.info(f"  {param}: {value}")

    # Evaluate on test set
    best_model = search.best_estimator_
    y_test_pred = best_model.predict(X_test)

    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    logger.info(f"\n{'='*60}")
    logger.info("TEST SET PERFORMANCE")
    logger.info("="*60)
    logger.info(f"  R² Score: {test_r2:.4f}")
    logger.info(f"  RMSE: ${test_rmse:.4f}")
    logger.info(f"  MAE: ${test_mae:.4f}")

    # Show top 5 parameter combinations
    logger.info(f"\n{'='*60}")
    logger.info("TOP 5 PARAMETER COMBINATIONS")
    logger.info("="*60)

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results['mean_test_rmse'] = np.sqrt(-cv_results['mean_test_score'])
    cv_results = cv_results.sort_values('mean_test_score', ascending=False)

    for idx, row in cv_results.head(5).iterrows():
        logger.info(f"\nRank {cv_results.index.get_loc(idx) + 1}:")
        logger.info(f"  RMSE: ${row['mean_test_rmse']:.4f}")
        logger.info(f"  Parameters: {row['params']}")

    # Save best model
    logger.info(f"\n{'='*60}")
    logger.info("SAVING MODEL")
    logger.info("="*60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, output_path)
    logger.success(f"Best model saved to: {output_path}")

    # Save search results
    results_path = output_path.parent / f"{output_path.stem}_cv_results.csv"
    cv_results.to_csv(results_path, index=False)
    logger.info(f"CV results saved to: {results_path}")

    # Save best parameters as pickle (for easy loading)
    params_path = output_path.parent / f"{output_path.stem}_best_params.pkl"
    params_dict = {
        'best_params': search.best_params_,
        'test_metrics': {
            'r2': test_r2,
            'rmse': test_rmse,
            'mae': test_mae
        },
        'cv_score': -search.best_score_
    }
    joblib.dump(params_dict, params_path)
    logger.success(f"Best parameters saved to: {params_path}")

    # Also save human-readable version
    params_txt_path = output_path.parent / f"{output_path.stem}_best_params.txt"
    with open(params_txt_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("BEST RANDOM FOREST PARAMETERS\n")
        f.write("="*60 + "\n\n")
        for param, value in search.best_params_.items():
            f.write(f"{param}: {value}\n")
        f.write(f"\nTest R² Score: {test_r2:.4f}\n")
        f.write(f"Test RMSE: ${test_rmse:.4f}\n")
        f.write(f"Test MAE: ${test_mae:.4f}\n")
    logger.info(f"Human-readable parameters saved to: {params_txt_path}")

    logger.info(f"\n{'='*60}")
    logger.success("HYPERPARAMETER TUNING COMPLETE!")
    logger.info("="*60)


@app.command()
def compare_default_vs_tuned(
    data_path: Path = PROCESSED_DATA_DIR / "rice_prices_cleaned.csv",
    tuned_model_path: Path = MODELS_DIR / "rf_tuned_model.pkl",
    test_size: float = 0.2,
):
    """
    Compare default Random Forest vs tuned model.

    Parameters:
    -----------
    data_path : Path
        Path to cleaned dataset
    tuned_model_path : Path
        Path to tuned model
    test_size : float
        Test set proportion
    """
    logger.info("="*60)
    logger.info("COMPARING DEFAULT VS TUNED RANDOM FOREST")
    logger.info("="*60)

    # Load data
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test, _ = preprocess_data(df, test_size=test_size)

    # Train default model
    logger.info("\nTraining default Random Forest...")
    default_rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    default_rf.fit(X_train, y_train)

    # Load tuned model
    logger.info("Loading tuned model...")
    tuned_rf = joblib.load(tuned_model_path)

    # Evaluate both
    models = {
        'Default': default_rf,
        'Tuned': tuned_rf
    }

    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON RESULTS")
    logger.info("="*60)

    for name, model in models.items():
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        logger.info(f"\n{name} Model:")
        logger.info(f"  R² Score: {r2:.4f}")
        logger.info(f"  RMSE: ${rmse:.4f}")
        logger.info(f"  MAE: ${mae:.4f}")

    # Calculate improvement
    default_r2 = r2_score(y_test, default_rf.predict(X_test))
    tuned_r2 = r2_score(y_test, tuned_rf.predict(X_test))
    improvement = ((tuned_r2 - default_r2) / default_r2) * 100

    logger.info(f"\n{'='*60}")
    if improvement > 0:
        logger.success(f"Improvement: +{improvement:.2f}% in R² score")
    else:
        logger.warning(f"Change: {improvement:.2f}% in R² score")


if __name__ == "__main__":
    app()
