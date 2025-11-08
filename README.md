# Rice Price Prediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

ML Model to predict rice prices in a given region based on historic prices using a stacking ensemble approach.

## Overview

This project implements a production-ready machine learning system for predicting rice prices per kilogram (USD) using:
- **Stacking Ensemble Model** combining Linear Regression, XGBoost, LightGBM, CatBoost, and Random Forest
- **FastAPI REST API** for serving predictions
- **Docker containerization** for easy deployment
- **Hyperparameter tuning** with GridSearchCV

### Model Performance

- **Test R² Score**: ~0.73
- **Features**: Country, location, commodity type, market type, temporal data
- **Target**: Price per kg (USD)

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd rpp

# Install dependencies
pip install -e .
```

### 2. Data Processing

```bash
# Process raw data
python -m rice_price_prediction.dataset

# Generate visualizations
python -m rice_price_prediction.plots
```

### 3. Model Training

```bash
# Train the ensemble model (with tuned Random Forest)
python -m rice_price_prediction.modeling.train

# Optional: Tune Random Forest hyperparameters
python -m rice_price_prediction.modeling.tune_rf tune
```

### 4. Run API Server

#### Option A: Local Development
```bash
fastapi dev rice_price_prediction/api/main.py
```

#### Option B: Docker (Recommended)
```bash
# Start the containerized API
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

API will be available at:
- **Interactive Docs**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/health

## Project Structure

```
├── LICENSE
├── README.md                  <- This file
├── DOCKER_README.md          <- Docker deployment guide
├── Dockerfile                <- Multi-stage Docker build
├── docker-compose.yml        <- Docker Compose configuration
├── .dockerignore             <- Docker build exclusions
├── pyproject.toml            <- Project dependencies and metadata
│
├── data/
│   ├── raw/                  <- Original, immutable data
│   └── processed/            <- Cleaned and transformed data
│
├── models/                   <- Trained models (.pkl files)
│   ├── stacking_ensemble_model.pkl
│   ├── label_encoders.pkl
│   └── rf_tuned_model.pkl
│
├── notebooks/                <- Jupyter notebooks for exploration
│   ├── eda.ipynb            <- Exploratory data analysis
│   └── process.ipynb        <- Data processing experiments
│
└── rice_price_prediction/   <- Source code
    ├── __init__.py
    ├── config.py             <- Configuration and paths
    ├── dataset.py            <- Data loading and cleaning
    ├── features.py           <- Feature engineering
    ├── plots.py              <- Visualization utilities
    │
    ├── api/                  <- FastAPI application
    │   ├── main.py           <- API endpoints
    │   ├── save_model.py     <- Model persistence utilities
    │   └── test.py           <- API test script
    │
    └── modeling/             <- ML pipeline
        ├── train.py          <- Stacking ensemble training
        ├── tune_rf.py        <- Random Forest hyperparameter tuning
        └── predict.py        <- Inference utilities
```

## API Usage

### Health Check
```bash
curl http://localhost:8001/health
```

### Make Prediction
```bash
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "country_code": "BGD",
    "county": "Dhaka",
    "subcounty": "Dhaka",
    "latitude": 23.81,
    "longitude": 90.41,
    "commodity": "Rice (coarse, BR-8/ 11/, Guti Sharna)",
    "price_flag": "actual",
    "price_type": "Wholesale",
    "year": 2024,
    "month": 11,
    "day": 7
  }'
```

**Response:**
```json
{
  "predicted_price_per_kg_usd": 0.85,
  "model_type": "Stacking Ensemble (LinearReg + XGBoost + LightGBM + CatBoost + RandomForest)"
}
```

## Model Architecture

### Stacking Ensemble

**Base Models (Level 0):**
- Linear Regression
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor
- Random Forest Regressor (hyperparameter tuned)

**Meta-Model (Level 1):**
- Ridge Regression

### Feature Engineering

- **Categorical Encoding**: Label encoding for categorical features
- **Temporal Features**: Year, month, day extracted from date
- **Removed Leakage**: Price-related features excluded from training


## Development Workflow

### 1. Complete ML Pipeline

```bash
# Process data
python -m rice_price_prediction.dataset

# Train model with hyperparameter tuning
python -m rice_price_prediction.modeling.tune_rf tune
python -m rice_price_prediction.modeling.train

# Generate analysis plots
python -m rice_price_prediction.plots
```

### 2. Run Tests

```bash
# Test API locally
python rice_price_prediction/api/test.py

# Or use pytest (if tests are added)
pytest tests/
```

### 3. Docker Development

```bash
# Rebuild after code changes
docker-compose up -d --build

# View logs
docker-compose logs -f api

# Access container shell
docker-compose exec api /bin/bash
```

## Docker Deployment

See [DOCKER_README.md](DOCKER_README.md) for detailed Docker deployment instructions.

**Key Features:**
- Multi-stage build (optimized image size: ~1.3GB)
- Non-root user for security
- Health checks included
- Volume mounts for models (industry standard)
- Resource limits configured

## Configuration

Edit `rice_price_prediction/config.py` to customize:
- Data paths
- Model paths
- Feature columns
- Target variable

## Technologies Used

- **ML**: scikit-learn, XGBoost, LightGBM, CatBoost
- **API**: FastAPI, Uvicorn
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Deployment**: Docker, Docker Compose
- **Development**: Jupyter, loguru, typer

## Model Tuning

### Random Forest Hyperparameter Tuning

```bash
# Run grid search (takes ~10-20 minutes)
python -m rice_price_prediction.modeling.tune_rf tune

# Compare default vs tuned
python -m rice_price_prediction.modeling.tune_rf compare-default-vs-tuned
```

**Tuning Results:**
- Parameters saved to: `models/rf_tuned_model_best_params.pkl`
- CV results: `models/rf_tuned_model_cv_results.csv`
- Human-readable: `models/rf_tuned_model_best_params.txt`

## Data Processing

**Input:** Global WFP Food Prices dataset
**Processing Steps:**
1. Filter for rice commodities
2. Remove duplicates
3. Convert units to standardized kg
4. Calculate price per kg (USD)
5. Remove outliers (>$5/kg)
6. Handle missing values

**Output:** `data/processed/rice_prices_cleaned.csv`

## Acknowledgments

- Global WFP Food Prices dataset
- Cookiecutter Data Science template

