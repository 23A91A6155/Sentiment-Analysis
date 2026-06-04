"""
Tests for the Sentiment Analysis FastAPI backend.

Uses pytest + httpx (via FastAPI's TestClient) to exercise
the /health and /predict endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from src.api import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """
    Provide a TestClient that runs the full application lifespan.

    If the model is not available on disk the lifespan handler sets
    app.state.model / app.state.tokenizer to None, so model-dependent
    tests can be conditionally skipped.
    """
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def model_available(client):
    """Return True when the model was loaded successfully at startup."""
    model = getattr(app.state, "model", None)
    tokenizer = getattr(app.state, "tokenizer", None)
    return model is not None and tokenizer is not None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_endpoint(self, client):
        """GET /health returns 200 with {"status": "ok"}."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data == {"status": "ok"}


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_valid_input(self, client, model_available):
        """POST /predict with valid text returns 200, sentiment, and confidence."""
        if not model_available:
            pytest.skip("Model not loaded – skipping inference test.")

        response = client.post("/predict", json={"text": "I love this product!"})
        assert response.status_code == 200

        data = response.json()
        assert "sentiment" in data
        assert "confidence" in data
        assert data["sentiment"] in ("positive", "negative")
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_empty_text(self, client):
        """POST /predict with empty text returns 400."""
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 400

        data = response.json()
        assert data["detail"] == "Text field cannot be empty"

    def test_predict_whitespace_text(self, client):
        """POST /predict with whitespace-only text returns 400."""
        response = client.post("/predict", json={"text": "   "})
        assert response.status_code == 400

        data = response.json()
        assert data["detail"] == "Text field cannot be empty"

    def test_predict_invalid_body(self, client):
        """POST /predict with wrong schema returns 422."""
        response = client.post("/predict", json={"wrong_key": 123})
        assert response.status_code == 422
