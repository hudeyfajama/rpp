"""
Feature engineering module for rice price prediction.

This module handles encoding categorical variables and preparing features
for machine learning models.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import typer

from rice_price_prediction.config import PROCESSED_DATA_DIR

app = typer.Typer()


def preprocess_data(
    df: pd.DataFrame,
    target_col: str = 'price_per_kg_usd',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, LabelEncoder]]:
    """
    Preprocess data for machine learning models.

    This function:
    1. Encodes categorical variables using LabelEncoder
    2. Converts date column to numerical features (year, month, day)
    3. Splits data into train/test sets

    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    target_col : str
        Name of target column (default: 'price_per_kg_usd')
    test_size : float
        Proportion of data for test set (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)

    Returns:
    --------
    X_train : DataFrame
        Training features
    X_test : DataFrame
        Test features
    y_train : Series
        Training target
    y_test : Series
        Test target
    label_encoders : dict
        Dictionary of fitted label encoders for each categorical column
    """
    logger.info("Starting data preprocessing...")

    # Define leaky columns (DO NOT include target_col here!)
    leaky_columns = [
        'price_usd',         # Leakage: derived from target
        'local_price',       # Leakage: target in different currency
        'kg',                # Leakage: used to calculate target (price_usd / kg = price_per_kg_usd)
        'market_id',         # High cardinality ID (not useful)
        'commodity_id',      # High cardinality ID (not useful)
        'category',          # Only one value (all cereals and tubers)
        'unit',              # Already converted to kg
        'currency',          # Already converted to USD
    ]

    # Only drop columns that exist
    cols_to_drop = [col for col in leaky_columns if col in df.columns]

    logger.info(f"Dropping leaky/unnecessary columns: {cols_to_drop}")

    # Separate features and target
    # First get target, then drop it + leaky columns from features
    y = df[target_col]
    X = df.drop(columns=[target_col] + cols_to_drop)

    # Encode categorical variables
    X_encoded = X.copy()
    label_encoders = {}

    categorical_cols = X.select_dtypes(include=['object']).columns
    logger.info(f"Encoding {len(categorical_cols)} categorical features...")

    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        logger.debug(f"  Encoded '{col}': {len(le.classes_)} unique values")

    # Convert date to numerical features
    if 'date' in X_encoded.columns:
        logger.info("Converting date to numerical features...")
        X_encoded['date'] = pd.to_datetime(X_encoded['date'])
        X_encoded['year'] = X_encoded['date'].dt.year
        X_encoded['month'] = X_encoded['date'].dt.month
        X_encoded['day'] = X_encoded['date'].dt.day
        X_encoded = X_encoded.drop(columns=['date'])
        logger.info("  Created: year, month, day features")

    # Split data into training and testing sets
    logger.info(f"Splitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state
    )

    logger.success("Preprocessing complete!")
    logger.info(f"Features: {list(X_train.columns)}")
    logger.info(f"Training set: {len(X_train):,} samples")
    logger.info(f"Test set: {len(X_test):,} samples")

    return X_train, X_test, y_train, y_test, label_encoders


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "rice_prices_cleaned.csv",
    output_dir: Path = PROCESSED_DATA_DIR,
    test_size: float = 0.2,
):
    """
    Load data, engineer features, and save train/test splits.

    Parameters:
    -----------
    input_path : Path
        Path to cleaned dataset
    output_dir : Path
        Directory to save processed features
    test_size : float
        Proportion for test set (default: 0.2)
    """
    logger.info("="*60)
    logger.info("FEATURE ENGINEERING")
    logger.info("="*60)

    # Load data
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns")

    # Preprocess and split
    X_train, X_test, y_train, y_test, label_encoders = preprocess_data(
        df, test_size=test_size
    )

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving processed features to {output_dir}")
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False, header=True)
    y_test.to_csv(output_dir / "y_test.csv", index=False, header=True)

    # Save label encoders
    import joblib
    encoders_path = output_dir / "label_encoders.pkl"
    joblib.dump(label_encoders, encoders_path)
    logger.info(f"Saved label encoders to {encoders_path}")

    logger.success("Feature engineering complete!")
    logger.info(f"Train/test data saved to {output_dir}")


if __name__ == "__main__":
    app()
