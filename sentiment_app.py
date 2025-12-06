import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🧠",
    layout="centered"
)

# 2. LOAD MODEL (With Caching)
# We use @st.cache_resource so it downloads the model only once
@st.cache_resource
def load_model():
    model_path = "shahbazahmadshahbazahmad/bert-finetuned-encoder"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        return None, str(e)

# Load the model immediately when app starts
with st.spinner("Downloading Model from Hugging Face... This may take a minute..."):
    tokenizer, model = load_model()

# 3. SENTIMENT MAPPING
# Ensure this matches your training labels!
labels_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# 4. USER INTERFACE
st.title("🧠 AI Sentiment Analyzer")
st.markdown("Type a sentence below, and the AI will detect if the emotion is **Positive**, **Negative**, or **Neutral**.")

# Input Box
user_input = st.text_area("Enter text here:", height=150, placeholder="e.g., The food was amazing but the service was slow.")

# Analyze Button
if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    elif isinstance(model, str): # Check if model failed to load
        st.error(f"Error loading model: {model}")
    else:
        with st.spinner("Analyzing..."):
            # A. Tokenize
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
            
            # B. Predict
            with torch.no_grad():
                outputs = model(**inputs)
            
            # C. Get the highest probability class
            logits = outputs.logits
            prediction_id = logits.argmax().item()
            sentiment = labels_map.get(prediction_id, "Unknown")
            
            # D. Display Result with Colors
            st.divider()
            st.subheader("Analysis Result:")
            
            if prediction_id == 0: # Negative
                st.error(f"😡 **{sentiment}**")
            elif prediction_id == 2: # Positive
                st.success(f"😃 **{sentiment}**")
            else: # Neutral (1)
                st.info(f"😐 **{sentiment}**")

# Sidebar Info
st.sidebar.info("Model: BERT Finetuned Encoder")
st.sidebar.text("Task: Sentiment Classification")
st.sidebar.markdown("[View on Hugging Face](https://huggingface.co/shahbazahmadshahbazahmad/bert-finetuned-encoder)")