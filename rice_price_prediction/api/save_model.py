"""
Script to save trained ensemble model and label encoders for API deployment.

This should be run after training your stacking ensemble model in the notebook.
"""

import joblib
from pathlib import Path
from loguru import logger

from rice_price_prediction.config import MODELS_DIR


def save_model(model, label_encoders, model_name="stacking_ensemble_model"):
    """
    Save the trained model and label encoders to disk.

    Parameters:
    -----------
    model : sklearn model
        Trained model to save
    label_encoders : dict
        Dictionary of label encoders
    model_name : str
        Name for the model file (without extension)
    """
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = MODELS_DIR / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    logger.success(f"Model saved to {model_path}")

    # Save label encoders
    encoders_path = MODELS_DIR / "label_encoders.pkl"
    joblib.dump(label_encoders, encoders_path)
    logger.success(f"Label encoders saved to {encoders_path}")

    return model_path, encoders_path


if __name__ == "__main__":
    logger.info("This script should be imported and used in your notebook.")
    logger.info("Example usage:")
    logger.info("  from rice_price_prediction.api.save_model import save_model")
    logger.info("  save_model(stacking_model, label_encoders)")
