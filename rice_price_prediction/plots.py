"""
Plotting and visualization module for rice price prediction.

This module provides functions for exploratory data analysis and
model performance visualization.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
import typer

from rice_price_prediction.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def plot_price_distribution(df: pd.DataFrame, output_path: Path = None):
    """
    Plot the distribution of rice prices.

    Parameters:
    -----------
    df : DataFrame
        Data containing 'price_per_kg_usd' column
    output_path : Path, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Kurtosis
    df.kurtosis(numeric_only=True).plot(kind='bar', ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('Kurtosis of Numerical Features')
    axes[0, 0].set_ylabel('Kurtosis')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Skewness
    df.skew(numeric_only=True).plot(kind='bar', ax=axes[0, 1], color='coral')
    axes[0, 1].set_title('Skewness of Numerical Features')
    axes[0, 1].set_ylabel('Skewness')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Histogram of price_per_kg_usd
    axes[1, 0].hist(df['price_per_kg_usd'], bins=50, edgecolor='black', color='lightgreen')
    axes[1, 0].set_title('Distribution of Price per KG (USD)')
    axes[1, 0].set_xlabel('Price per KG (USD)')
    axes[1, 0].set_ylabel('Frequency')

    # 4. Histogram with log scale
    axes[1, 1].hist(df['price_per_kg_usd'], bins=50, edgecolor='black', color='lightblue')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('Distribution of Price per KG (USD) - Log Scale')
    axes[1, 1].set_xlabel('Price per KG (USD)')
    axes[1, 1].set_ylabel('Frequency (log scale)')

    plt.suptitle('Statistical Analysis of Numerical Features', fontsize=16, y=1.00)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_price_by_category(df: pd.DataFrame, column: str, output_path: Path = None):
    """
    Plot rice prices grouped by a categorical column.

    Parameters:
    -----------
    df : DataFrame
        Data containing price and categorical columns
    column : str
        Column name to group by
    output_path : Path, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Box plot
    df.boxplot(column='price_per_kg_usd', by=column, ax=axes[0])
    axes[0].set_title(f'Price Distribution by {column}')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Price per KG (USD)')
    plt.sca(axes[0])
    plt.xticks(rotation=45)

    # Bar plot of mean prices
    mean_prices = df.groupby(column)['price_per_kg_usd'].mean().sort_values(ascending=False).head(20)
    mean_prices.plot(kind='barh', ax=axes[1], color='teal')
    axes[1].set_title(f'Average Price by {column} (Top 20)')
    axes[1].set_xlabel('Average Price per KG (USD)')
    axes[1].set_ylabel(column)

    plt.suptitle(f'Price Analysis by {column}', fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_price_over_time(df: pd.DataFrame, output_path: Path = None):
    """
    Plot rice prices over time.

    Parameters:
    -----------
    df : DataFrame
        Data containing 'date' and 'price_per_kg_usd' columns
    output_path : Path, optional
        Path to save the plot
    """
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'])

    # Group by month and calculate mean
    df_monthly = df_copy.groupby(pd.Grouper(key='date', freq='M'))['price_per_kg_usd'].agg(['mean', 'std', 'count'])

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Line plot with confidence interval
    axes[0].plot(df_monthly.index, df_monthly['mean'], color='blue', linewidth=2)
    axes[0].fill_between(
        df_monthly.index,
        df_monthly['mean'] - df_monthly['std'],
        df_monthly['mean'] + df_monthly['std'],
        alpha=0.3, color='blue'
    )
    axes[0].set_title('Average Rice Price Over Time (Monthly)')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price per KG (USD)')
    axes[0].grid(True, alpha=0.3)

    # Number of records over time
    axes[1].bar(df_monthly.index, df_monthly['count'], color='lightcoral', width=20)
    axes[1].set_title('Number of Price Records Over Time')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Count')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, output_path: Path = None):
    """
    Plot correlation matrix of numerical features.

    Parameters:
    -----------
    df : DataFrame
        Data with numerical columns
    output_path : Path, optional
        Path to save the plot
    """
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr = df[numerical_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "rice_prices_cleaned.csv",
    output_dir: Path = FIGURES_DIR,
):
    """
    Generate exploratory data analysis plots.

    Parameters:
    -----------
    input_path : Path
        Path to processed dataset
    output_dir : Path
        Directory to save plots
    """
    logger.info("="*60)
    logger.info("GENERATING EDA PLOTS")
    logger.info("="*60)

    # Load data
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} records")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    logger.info("\nGenerating price distribution plots...")
    plot_price_distribution(df, output_dir / "price_distribution.png")

    logger.info("Generating price by country plots...")
    plot_price_by_category(df, 'country_code', output_dir / "price_by_country.png")

    logger.info("Generating price by commodity plots...")
    plot_price_by_category(df, 'commodity', output_dir / "price_by_commodity.png")

    logger.info("Generating price over time plots...")
    plot_price_over_time(df, output_dir / "price_over_time.png")

    logger.info("Generating correlation matrix...")
    plot_correlation_matrix(df, output_dir / "correlation_matrix.png")

    logger.success(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    app()
