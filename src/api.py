"""
FastAPI backend for Sentiment Analysis.

Provides endpoints for health checking and sentiment prediction
using a fine-tuned transformer model.
"""

import logging
import os
from contextlib import asynccontextmanager

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH: str = os.getenv("MODEL_PATH", "model_output")
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Schema for the prediction request body."""
    text: str


class PredictResponse(BaseModel):
    """Schema for the prediction response body."""
    sentiment: str
    confidence: float


class HealthResponse(BaseModel):
    """Schema for the health-check response."""
    status: str


# ---------------------------------------------------------------------------
# Lifespan – load model & tokenizer once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load model and tokenizer into app.state at startup."""
    try:
        logger.info("Loading tokenizer from '%s' …", MODEL_PATH)
        application.state.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        logger.info("Loading model from '%s' …", MODEL_PATH)
        application.state.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
        )
        application.state.model.eval()

        logger.info("Model and tokenizer loaded successfully.")
    except Exception:
        logger.exception("Failed to load model/tokenizer – endpoints that require the model will return 503.")
        application.state.model = None
        application.state.tokenizer = None

    yield  # application runs

    # Cleanup (if needed)
    logger.info("Shutting down – releasing model resources.")
    application.state.model = None
    application.state.tokenizer = None


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Sentiment Analysis API",
    description="Predict sentiment (positive / negative) for a given text.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, status_code=200)
async def health():
    """Simple liveness / readiness probe."""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse, status_code=200)
async def predict(request: PredictRequest):
    """
    Run sentiment inference on the supplied text.

    Returns the predicted sentiment label and the model's confidence score.
    """
    # --- Validate input --------------------------------------------------- #
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty")

    # --- Check model availability ----------------------------------------- #
    model = getattr(app.state, "model", None)
    tokenizer = getattr(app.state, "tokenizer", None)

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Please ensure the model is loaded before making predictions.",
        )

    # --- Inference -------------------------------------------------------- #
    try:
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

        sentiment = "positive" if predicted_class == 1 else "negative"
        confidence = round(confidence, 4)

        return PredictResponse(sentiment=sentiment, confidence=confidence)

    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error during prediction.")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the request.",
        )


# ---------------------------------------------------------------------------
# Run with: python src/api.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("API_HOST", "0.0.0.0"), port=int(os.getenv("API_PORT", "8000")))
