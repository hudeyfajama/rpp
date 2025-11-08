"""
Model training module for rice price prediction.

This module implements a stacking ensemble model combining multiple
regression algorithms for robust price predictions.
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from loguru import logger
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import typer

from rice_price_prediction.config import MODELS_DIR, PROCESSED_DATA_DIR
from rice_price_prediction.features import preprocess_data

app = typer.Typer()


def load_rf_best_params() -> dict:
    """
    Load best Random Forest parameters from tuning.

    Returns:
    --------
    dict
        Best parameters from hyperparameter tuning, or default params if not found
    """
    params_path = MODELS_DIR / "rf_tuned_model_best_params.pkl"

    if params_path.exists():
        logger.info(f"Loading tuned RF parameters from {params_path}")
        params_dict = joblib.load(params_path)
        params = params_dict['best_params'].copy()

        logger.success(f"Loaded tuned parameters: {params}")
        logger.info(f"Tuning test metrics - R²: {params_dict['test_metrics']['r2']:.4f}, "
                   f"RMSE: ${params_dict['test_metrics']['rmse']:.4f}")
        return params
    else:
        logger.warning(f"Tuned parameters not found at {params_path}")
        logger.warning("Using default Random Forest parameters")
        return {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt'
        }


def create_stacking_model(use_tuned_rf: bool = True) -> StackingRegressor:
    """
    Create a stacking ensemble model with multiple base estimators.

    Parameters:
    -----------
    use_tuned_rf : bool
        Whether to use tuned Random Forest parameters (default: True)

    Returns:
    --------
    StackingRegressor
        Configured stacking ensemble model
    """
    # Get Random Forest parameters
    if use_tuned_rf:
        rf_params = load_rf_best_params()
        # Ensure required parameters are set
        rf_params['random_state'] = 42
        rf_params['n_jobs'] = -1
        rf_params['verbose'] = 0
    else:
        rf_params = {
            'n_estimators': 100,
            'max_depth': 15,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 0
        }

    # Define base models (Level 0)
    base_models = [
        ('linear_reg', LinearRegression()),
        ('xgboost', xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )),
        ('lightgbm', lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )),
        ('catboost', CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=False
        )),
        ('random_forest', RandomForestRegressor(**rf_params))
    ]

    # Define meta-model (Level 1)
    meta_model = Ridge(alpha=1.0)

    # Create stacking ensemble
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,  # 5-fold cross-validation for base model predictions
        n_jobs=-1
    )

    return stacking_model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model performance on train and test sets.

    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X_train, X_test : DataFrame
        Training and test features
    y_train, y_test : Series
        Training and test targets

    Returns:
    --------
    dict
        Dictionary containing all metrics
    """
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'train': {
            'r2': r2_score(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'mae': mean_absolute_error(y_train, y_train_pred),
        },
        'test': {
            'r2': r2_score(y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'mae': mean_absolute_error(y_test, y_test_pred),
        }
    }

    return metrics


def print_metrics(metrics: dict):
    """Print formatted model metrics."""
    logger.info("="*60)
    logger.info("MODEL PERFORMANCE")
    logger.info("="*60)

    logger.info("\nTraining Set Metrics:")
    logger.info(f"  R² Score: {metrics['train']['r2']:.4f}")
    logger.info(f"  RMSE: ${metrics['train']['rmse']:.4f}")
    logger.info(f"  MAE: ${metrics['train']['mae']:.4f}")

    logger.info("\nTest Set Metrics:")
    logger.info(f"  R² Score: {metrics['test']['r2']:.4f}")
    logger.info(f"  RMSE: ${metrics['test']['rmse']:.4f}")
    logger.info(f"  MAE: ${metrics['test']['mae']:.4f}")


@app.command()
def main(
    data_path: Path = PROCESSED_DATA_DIR / "rice_prices_cleaned.csv",
    model_output_path: Path = MODELS_DIR / "stacking_ensemble_model.pkl",
    encoders_output_path: Path = MODELS_DIR / "label_encoders.pkl",
    test_size: float = 0.2,
    use_tuned_rf: bool = typer.Option(True, help="Use tuned Random Forest parameters"),
):
    """
    Train stacking ensemble model for rice price prediction.

    Parameters:
    -----------
    data_path : Path
        Path to cleaned dataset
    model_output_path : Path
        Path to save trained model
    encoders_output_path : Path
        Path to save label encoders
    test_size : float
        Proportion for test set (default: 0.2)
    use_tuned_rf : bool
        Whether to use tuned Random Forest parameters
    """
    logger.info("="*60)
    logger.info("RICE PRICE PREDICTION - MODEL TRAINING")
    logger.info("="*60)

    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df):,} records")

    # Preprocess data
    logger.info("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, label_encoders = preprocess_data(
        df, test_size=test_size
    )

    # Create model
    logger.info("\n" + "="*60)
    logger.info("CREATING STACKING ENSEMBLE MODEL")
    logger.info("="*60)
    logger.info("Base Models:")
    logger.info("  - Linear Regression")
    logger.info("  - XGBoost")
    logger.info("  - LightGBM")
    logger.info("  - CatBoost")
    logger.info(f"  - Random Forest ({'TUNED' if use_tuned_rf else 'DEFAULT'})")
    logger.info("\nMeta-Model: Ridge Regression")

    stacking_model = create_stacking_model(use_tuned_rf=use_tuned_rf)

    # Train model
    logger.info("\n" + "="*60)
    logger.info("TRAINING MODEL")
    logger.info("="*60)
    logger.info("This may take a few minutes...")

    stacking_model.fit(X_train, y_train)
    logger.success("Training complete!")

    # Evaluate model
    logger.info("\nEvaluating model...")
    metrics = evaluate_model(stacking_model, X_train, X_test, y_train, y_test)
    print_metrics(metrics)

    # Save model and encoders
    logger.info("\n" + "="*60)
    logger.info("SAVING MODEL AND ENCODERS")
    logger.info("="*60)

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    logger.info(f"Saving model to {model_output_path}")
    joblib.dump(stacking_model, model_output_path)
    logger.success(f"Model saved: {model_output_path}")

    # Save label encoders
    logger.info(f"Saving label encoders to {encoders_output_path}")
    joblib.dump(label_encoders, encoders_output_path)
    logger.success(f"Label encoders saved: {encoders_output_path}")

    logger.info("\n" + "="*60)
    logger.success("MODEL TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Test R² Score: {metrics['test']['r2']:.4f}")
    logger.info(f"Test RMSE: ${metrics['test']['rmse']:.4f}")
    logger.info(f"Test MAE: ${metrics['test']['mae']:.4f}")


if __name__ == "__main__":
    app()
