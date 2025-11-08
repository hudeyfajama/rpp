from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from rice_price_prediction.config import MODELS_DIR

# Initialize FastAPI app
app = FastAPI(
    title="Rice Price Prediction API",
    description="API for predicting rice prices using ensemble machine learning models",
    version="1.0.0"
)

# Global variable to store loaded model
model = None
label_encoders = None


class PredictionRequest(BaseModel):
    """Request model for price prediction"""
    country_code: str = Field(..., description="Country code (e.g., 'BGD')")
    county: str = Field(..., description="County name")
    subcounty: str = Field(..., description="Subcounty name")
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    commodity: str = Field(..., description="Rice commodity type")
    price_flag: str = Field(..., description="Price flag (e.g., 'actual')")
    price_type: str = Field(..., description="Price type (e.g., 'Wholesale', 'Retail')")
    year: int = Field(..., description="Year", ge=2000, le=2100)
    month: int = Field(..., description="Month", ge=1, le=12)
    day: int = Field(..., description="Day", ge=1, le=31)

    class Config:
        json_schema_extra = {
            "example": {
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
            }
        }


class PredictionResponse(BaseModel):
    """Response model for price prediction"""
    predicted_price_per_kg_usd: float = Field(..., description="Predicted price per kg in USD")
    model_type: str = Field(..., description="Type of model used")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    model_path: Optional[str] = None


@app.on_event("startup")
async def load_model():
    """Load the trained model on startup"""
    global model, label_encoders

    model_path = MODELS_DIR / "stacking_ensemble_model.pkl"
    encoders_path = MODELS_DIR / "label_encoders.pkl"

    try:
        if model_path.exists():
            logger.info(f"Loading model from {model_path}")
            model = joblib.load(model_path)
            logger.success("Model loaded successfully")
        else:
            logger.warning(f"Model file not found at {model_path}")
            logger.warning("Model will need to be trained and saved first")

        if encoders_path.exists():
            logger.info(f"Loading label encoders from {encoders_path}")
            label_encoders = joblib.load(encoders_path)
            logger.success("Label encoders loaded successfully")
        else:
            logger.warning(f"Label encoders not found at {encoders_path}")

    except Exception as e:
        logger.error(f"Error loading model or encoders: {e}")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Rice Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_path = MODELS_DIR / "stacking_ensemble_model.pkl"

    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        model_path=str(model_path) if model_path.exists() else None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """
    Predict rice price based on input features

    Returns predicted price per kg in USD
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train and save the model first."
        )

    try:
        # Create DataFrame from request
        input_data = pd.DataFrame([{
            'country_code': request.country_code,
            'county': request.county,
            'subcounty': request.subcounty,
            'latitude': request.latitude,
            'longitude': request.longitude,
            'commodity': request.commodity,
            'price_flag': request.price_flag,
            'price_type': request.price_type,
            'year': request.year,
            'month': request.month,
            'day': request.day
        }])

        # Encode categorical variables if encoders are available
        if label_encoders:
            for col, encoder in label_encoders.items():
                if col in input_data.columns:
                    try:
                        input_data[col] = encoder.transform(input_data[col].astype(str))
                    except ValueError as e:
                        # Handle unseen categories
                        logger.warning(f"Unseen category in {col}: {input_data[col].values[0]}")
                        # Use the most frequent class as fallback
                        input_data[col] = encoder.transform([encoder.classes_[0]])[0]

        # Make prediction
        prediction = model.predict(input_data)[0]

        return PredictionResponse(
            predicted_price_per_kg_usd=float(prediction),
            model_type="Stacking Ensemble (LinearReg + XGBoost + LightGBM + CatBoost + RandomForest)"
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
