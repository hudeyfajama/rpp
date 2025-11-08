"""
Data processing module for rice price prediction.

This module handles data cleaning, unit conversion, and preprocessing
for the Global WFP Food Prices dataset.
"""

from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer

from rice_price_prediction.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def convert_to_kg(unit: str) -> float:
    """
    Convert various units to kilograms.

    Parameters:
    -----------
    unit : str
        Unit string to convert (e.g., "100 KG", "Pound", "50 KG")

    Returns:
    --------
    float
        Conversion factor to kilograms, or None if unit not recognized
    """
    unit = str(unit).strip()

    # Extract number if present (e.g., "100 KG" -> 100)
    parts = unit.split()
    if len(parts) == 2 and parts[0].replace('.', '').isdigit():
        quantity = float(parts[0])
        unit_type = parts[1].upper()
    else:
        quantity = 1
        unit_type = unit.upper()

    # Conversion factors to kilograms
    conversions = {
        'KG': 1,
        'G': 0.001,
        'MT': 1000,  # Metric Ton
        'POUND': 0.453592,
        'POUNDS': 0.453592,
        'LIBRA': 0.453592,  # Spanish/Portuguese pound
        'MARMITE': 2.7,  # not standardised - 6lb = 1 marmite
        'CUARTILLA': 2.875,  # Traditional Spanish unit, ~2.875 kg
    }

    # Get conversion factor
    for key, factor in conversions.items():
        if key in unit_type:
            return quantity * factor

    # If unit not recognized, return None
    return None


def process_rice_data(input_path: Path, max_price_per_kg: float = 5.0) -> pd.DataFrame:
    """
    Process raw food prices data to extract and clean rice prices.

    Parameters:
    -----------
    input_path : Path
        Path to raw CSV data file
    max_price_per_kg : float
        Maximum reasonable price per kg in USD (filters outliers)

    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe with standardized rice prices
    """
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} records")

    # Remove duplicates
    logger.info("Removing duplicates...")
    df = df.drop_duplicates()
    logger.info(f"After deduplication: {len(df):,} records")

    # Filter for rice commodities only
    logger.info("Filtering for rice commodities...")
    df = df[df['commodity'].str.contains('rice', case=False, na=False)]
    logger.info(f"Rice records: {len(df):,}")

    # Convert units to kilograms
    logger.info("Converting units to standardized kg...")
    df['kg'] = df['unit'].apply(convert_to_kg)

    # Check for unconverted units
    unconverted = df[df['kg'].isna()]
    if len(unconverted) > 0:
        logger.warning(f"Could not convert {len(unconverted)} records with units:")
        logger.warning(unconverted['unit'].value_counts().to_string())

    # Remove rows with invalid conversions
    df = df[df['kg'].notna()].copy()
    logger.info(f"After unit conversion: {len(df):,} records")

    # Calculate price per kilogram
    logger.info("Calculating price per kg...")
    df['price_per_kg_usd'] = df['price_usd'] / df['kg']

    # Filter outliers
    logger.info(f"Filtering prices > ${max_price_per_kg}/kg (outliers)...")
    df = df[df['price_per_kg_usd'] <= max_price_per_kg]
    logger.info(f"After outlier removal: {len(df):,} records")

    # Drop rows with missing values
    logger.info("Dropping rows with missing values...")
    df = df.dropna()
    logger.info(f"Final dataset: {len(df):,} records")

    logger.success("Data processing complete!")

    return df


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "Global WFP Food Prices.csv",
    output_path: Path = PROCESSED_DATA_DIR / "rice_prices_cleaned.csv",
    max_price_per_kg: float = 5.0,
):
    """
    Process raw food prices data and save cleaned rice prices.

    Parameters:
    -----------
    input_path : Path
        Path to raw CSV data file
    output_path : Path
        Path to save processed data
    max_price_per_kg : float
        Maximum reasonable price per kg in USD (default: 5.0)
    """
    logger.info("="*60)
    logger.info("RICE PRICE DATA PROCESSING")
    logger.info("="*60)

    # Process data
    df = process_rice_data(input_path, max_price_per_kg)

    # Save cleaned data
    logger.info(f"Saving processed data to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.success(f"File saved to: {output_path}")
    logger.success(f"Total records saved: {len(df):,}")
    logger.info(f"Dataframe shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    app()
