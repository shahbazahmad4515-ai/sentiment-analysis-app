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
        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load Model with memory optimizations
        # low_cpu_mem_usage=True requires the 'accelerate' library
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True
        )
        
        # Force CPU to ensure no GPU memory is requested
        device = torch.device("cpu")
        model.to(device)
        
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, str(e)

# Load the model (Cached)
with st.spinner("Downloading Model... (This may take a minute)"):
    tokenizer, model = load_model()

# 3. SENTIMENT MAPPING
labels_map = {
    0: "Negative",
    1: "Negative", # Treating Neutral as Negative per your logic
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
    # Check if model loaded successfully (it returns a string error if failed)
    elif isinstance(model, str) or model is None:
        st.error(f"Model failed to load. Error: {model}")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Tokenize
                inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
                
                # Run Inference on CPU
                with torch.no_grad():
                    outputs = model(**inputs)
                
                logits = outputs.logits
                prediction_id = logits.argmax().item()
                
                # Get label
                sentiment = labels_map.get(prediction_id, "Unknown")
                
                # Display Result
                st.divider()
                st.subheader("Analysis Result:")
                
                if prediction_id == 2:
                    st.success(f"😃 **{sentiment}**")
                else:
                    st.error(f"😡 **{sentiment}**")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

# Sidebar Info
st.sidebar.info("Model: BERT Finetuned")
st.sidebar.text("Classes: Positive / Negative")
st.sidebar.markdown("[View on Hugging Face](https://huggingface.co/shahbazahmadshahbazahmad/bert-finetuned-encoder)")
