import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🧠",
    layout="centered"
)

# 2. LOAD MODEL
@st.cache_resource
def load_model():
    model_path = "shahbazahmadshahbazahmad/bert-finetuned-encoder"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        return None, str(e)

with st.spinner("Downloading Model..."):
    tokenizer, model = load_model()

# 3. SENTIMENT MAPPING
# Based on your training snippet:
# 0 -> Negative
# 1 -> Neutral/Irrelevant
# 2 -> Positive
labels_map = {
    0: "Negative",
    1: "Negative", # We treat Neutral/Irrelevant as Negative to keep strict Binary output
    2: "Positive"
}

# 4. USER INTERFACE
st.title("🧠 AI Sentiment Analyzer")
st.markdown("Type a sentence below to detect if it is **Positive** or **Negative**." \
"It will show **Neutral** sentence as **Negative**")

# Input Box
user_input = st.text_area("Enter text here:", height=150, placeholder="e.g., The food was amazing!")

# Analyze Button
if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    elif isinstance(model, str):
        st.error(f"Error loading model: {model}")
    else:
        with st.spinner("Analyzing..."):
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            logits = outputs.logits
            prediction_id = logits.argmax().item()
            
            # Get label from our map
            sentiment = labels_map.get(prediction_id, "Unknown")
            
            # D. Display Result
            st.divider()
            st.subheader("Analysis Result:")
            
            # Logic: If it is Class 2, it is Positive (Green)
            if prediction_id == 2:
                st.success(f"😃 **{sentiment}**")
            
            # Logic: If it is Class 0 OR 1, it is Negative (Red)
            else:
                st.error(f"😡 **{sentiment}**")
                
# Sidebar Info
st.sidebar.info("Model: BERT Finetuned")
st.sidebar.text("Classes: Positive / Negative")
st.sidebar.markdown("[View on Hugging Face](https://huggingface.co/shahbazahmadshahbazahmad/bert-finetuned-encoder)")