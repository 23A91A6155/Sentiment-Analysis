"""
Streamlit Cloud Deployment — Sentiment Analysis System

Standalone version for cloud deployment that uses the
Hugging Face Inference API for predictions.
Same UI and functionality as src/ui.py.
"""

import streamlit as st
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

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
                    HF_API_URL,
                    json={"inputs": user_text},
                    timeout=30,
                )

                if response.status_code == 200:
                    data = response.json()

                    # HF API returns [[{"label": "POSITIVE", "score": 0.99}, ...]]
                    results = data[0] if isinstance(data[0], list) else data
                    top_result = max(results, key=lambda x: x["score"])

                    sentiment = top_result["label"].lower()
                    confidence = round(top_result["score"], 4)

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

                elif response.status_code == 503:
                    st.warning(
                        "⏳ Model is loading, please wait 20 seconds and try again."
                    )
                else:
                    st.error(
                        f"Unexpected response (HTTP {response.status_code}). "
                        "Please try again later."
                    )

            except requests.exceptions.ConnectionError:
                st.error("⚠️ Could not connect to the prediction service.")
            except requests.exceptions.Timeout:
                st.error("⚠️ The request timed out. Please try again.")
            except Exception as exc:
                st.error(f"⚠️ An error occurred: {exc}")
