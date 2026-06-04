"""
Streamlit UI for Sentiment Analysis.

Provides a clean, professional interface for users to enter text
and receive sentiment predictions from the FastAPI backend.
"""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL: str = os.getenv("API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analysis System",
    page_icon="🔍",
    layout="centered",
)

st.title("Sentiment Analysis System")
st.markdown("---")

# ---------------------------------------------------------------------------
# Input area
# ---------------------------------------------------------------------------
user_text: str = st.text_area(
    "Enter text for sentiment analysis:",
    height=200,
    placeholder="Type or paste the text you want to analyze…",
)

analyze_clicked: bool = st.button("Analyze Sentiment", type="primary")

# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------
if analyze_clicked:
    # Guard against empty / whitespace-only input
    if not user_text or not user_text.strip():
        st.warning("Please enter some text before clicking **Analyze Sentiment**.")
    else:
        with st.spinner("Analyzing sentiment…"):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": user_text},
                    timeout=30,
                )

                if response.status_code == 200:
                    data = response.json()
                    sentiment: str = data["sentiment"]
                    confidence: float = data["confidence"]

                    st.markdown("### Results")

                    # Sentiment with colour coding
                    if sentiment == "positive":
                        st.success(f"**Sentiment:** {sentiment.capitalize()} 😊")
                    else:
                        st.error(f"**Sentiment:** {sentiment.capitalize()} 😟")

                    # Confidence as a percentage metric
                    st.metric(
                        label="Confidence",
                        value=f"{confidence * 100:.2f}%",
                    )

                elif response.status_code == 400:
                    detail = response.json().get("detail", "Bad request.")
                    st.warning(detail)

                elif response.status_code == 503:
                    detail = response.json().get("detail", "Model not available.")
                    st.error(f"Service unavailable: {detail}")

                else:
                    st.error(
                        f"Unexpected response from the API (HTTP {response.status_code}). "
                        "Please try again later."
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "⚠️ Could not connect to the API. "
                    f"Please make sure the backend is running at **{API_URL}**."
                )
            except requests.exceptions.Timeout:
                st.error(
                    "⚠️ The request timed out. Please try again later."
                )
            except requests.exceptions.RequestException as exc:
                st.error(f"⚠️ An error occurred while contacting the API: {exc}")
