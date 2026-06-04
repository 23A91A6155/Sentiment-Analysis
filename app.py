"""
Streamlit Cloud Deployment — Sentiment Analysis System

Standalone version for cloud deployment that uses the
Hugging Face Inference API for predictions.
Same UI and functionality as src/ui.py.
"""

import streamlit as st
from huggingface_hub import InferenceClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

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
    if not user_text or not user_text.strip():
        st.warning("Please enter some text before clicking **Analyze Sentiment**.")
    else:
        with st.spinner("Analyzing sentiment…"):
            try:
                client = InferenceClient()
                results = client.text_classification(
                    user_text,
                    model=HF_MODEL,
                )

                # Get the top prediction
                top_result = results[0]
                sentiment = top_result.label.lower()
                confidence = round(top_result.score, 4)

                st.markdown("### Results")

                if sentiment == "positive":
                    st.success(f"**Sentiment:** {sentiment.capitalize()} 😊")
                else:
                    st.error(f"**Sentiment:** {sentiment.capitalize()} 😟")

                st.metric(
                    label="Confidence",
                    value=f"{confidence * 100:.2f}%",
                )

            except Exception as exc:
                st.error(f"⚠️ An error occurred during analysis: {exc}")
