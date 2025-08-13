import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re

# ==== SETTINGS ====
MODEL_PATH = "bert_model"
device = torch.device("cpu")  # Force CPU to avoid meta tensor issues

# ==== LOAD MODEL SAFELY ON CPU ====
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=False
).to(device)
model.eval()

# ==== TRUSTED SOURCES ====
trusted_sources = [
    "bbc.com", "reuters.com", "techxplore.com", "nytimes.com", "theguardian.com",
    "indiatoday.in", "hindustantimes.com", "ndtv.com", "timesofindia.com"
]

# ==== FUNCTIONS ====
def extract_domain(text):
    match = re.search(r"https?://(?:www\.)?([^/]+)", text)
    return match.group(1).lower() if match else None

def predict_news(text):
    # Tokenize and ensure inputs are on CPU
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        padding="max_length", max_length=256
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()

    # Apply source credibility adjustment
    domain = extract_domain(text)
    if domain in trusted_sources and pred_label == 0:
        if confidence < 0.80:
            pred_label = 1
            confidence = 0.85

    return pred_label, confidence, domain

# ==== STREAMLIT UI ====
st.set_page_config(page_title="Fake News Detector 2025", layout="centered")
st.title("📰 Fake News Detection (BERT + Source Check) - CPU Only")
st.subheader("Paste a news article or link to check if it's Real or Fake.")

user_input = st.text_area("Enter News Article Text or URL:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text or a URL.")
    else:
        label, conf, domain = predict_news(user_input)
        if label == 1:
            st.success(f"✅ Likely REAL (Confidence: {conf:.2%})")
        else:
            st.error(f"❌ Likely FAKE (Confidence: {conf:.2%})")

        if domain:
            st.info(f"Detected source: {domain}")
            if domain in trusted_sources:
                st.write("🔹 Source is trusted.")
            else:
                st.write("⚠️ Source not in trusted list.")
