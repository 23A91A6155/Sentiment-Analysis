"""
Streamlit Cloud Deployment — Sentiment Analysis System

Standalone version for cloud deployment that loads the model
directly instead of calling a separate API server.
Same UI and functionality as src/ui.py.
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analysis System",
    page_icon="🔍",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Load model (cached so it only loads once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load pre-trained sentiment analysis model and tokenizer."""
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
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
                inputs = tokenizer(
                    user_text,
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

            except Exception as exc:
                st.error(f"⚠️ An error occurred during analysis: {exc}")
