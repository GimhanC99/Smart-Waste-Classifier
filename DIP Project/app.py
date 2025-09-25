import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
import datetime
from tensorflow.keras.applications.resnet50 import preprocess_input

# CONFIG
MODEL_PATH = "resnet50_best.h5"
LOG_PATH = "Document.csv"
CLASSES = ['battery', 'glass', 'metal', 'organic', 'paper', 'plastic']

# Streamlit Page Settings
st.set_page_config(
    page_title="♻️ Smart Waste Classifier",
    page_icon="♻️",
    layout="wide"
)

# Custom CSS for professional UI
st.markdown("""
<style>
    /* General */
    .main { background-color: #f9f9f9; }
    h1, h2, h3, h4 { font-family: 'Segoe UI', sans-serif; }
    .stButton>button {
        border-radius: 12px;
        padding: 10px 20px;
        background-color: #2ecc71;
        color: white;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #27ae60;
    }
    /* Prediction Card */
    .pred-card {
        padding: 15px;
        margin: 12px 0;
        border-radius: 12px;
        background: #ffffff;
        border-left: 6px solid #2ecc71;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
    }
    .pred-title { font-size: 20px; font-weight: bold; margin-bottom: 5px; color: #145a32; }
    .pred-label { font-size: 22px; font-weight: bold; color: #27ae60; }
    /* Footer */
    .footer { text-align:center; margin-top: 30px; color: #777; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# HELPERS
def get_cache_decorator():
    try:
        return st.cache_resource
    except Exception:
        return st.cache(allow_output_mutation=True)

cache = get_cache_decorator()

@cache
def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model, CLASSES

def infer_target_size(model):
    try:
        shape = model.input_shape
        if shape and len(shape) == 4:
            h = shape[1] or 224
            w = shape[2] or 224
            return (int(h), int(w))
    except Exception:
        pass
    return (224, 224)

def preprocess_pil(img: Image.Image, target_size):
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

def append_log(label, log_path=LOG_PATH):
    now = datetime.datetime.now().isoformat()
    row = {"timestamp": now, "label": label}
    df = pd.DataFrame([row])
    header = not os.path.exists(log_path)
    df.to_csv(log_path, mode="a", header=header, index=False)

def load_log(log_path=LOG_PATH):
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=["timestamp", "label"])
    return pd.read_csv(log_path)

def build_report(classes, log_df):
    if "label" not in log_df.columns:
        return "No predictions yet."
    total = len(log_df)
    counts = log_df["label"].value_counts().to_dict()
    lines = ["Waste Classification Report", f"Total Items Processed: {total}"]
    for c in classes:
        lines.append(f"{c}: {counts.get(c, 0)}")
    return "\n".join(lines)

# ------------------- UI -------------------

st.title("♻️ Smart Waste Classifier")
st.write("Upload waste images to classify them using a trained CNN model. A persistent log is kept for reporting 📊.")

# Load Model
try:
    model, classes = load_model()
    target_size = infer_target_size(model)
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# File uploader
uploaded = st.file_uploader(
    "📂 Upload waste images (PNG, JPG, JPEG)", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

# Prediction Section
if uploaded:
    if st.button("🚀 Run Predictions"):
        results = []
        progress = st.progress(0)
        n = len(uploaded)
        for i, uf in enumerate(uploaded):
            try:
                img = Image.open(uf)
                x = preprocess_pil(img, target_size)
                preds = model.predict(x, verbose=0)
                idx = int(np.argmax(preds, axis=1)[0])
                label = classes[idx]
                append_log(label)
                results.append((uf.name, label, None))
            except Exception as err:
                results.append((uf.name, None, str(err)))
            progress.progress(int(((i+1)/n) * 100))
        st.success("✅ Predictions completed!")

        # Styled Prediction Cards
        for name, label, err in results:
            if err:
                st.error(f"❌ {name} → ERROR: {err}")
            else:
                st.markdown(
                    f"""
                    <div class="pred-card">
                        <div class="pred-title">📌 {name}</div>
                        <div class="pred-label">Prediction → {label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# Report Section
st.markdown("---")
st.header("📊 Waste Classification Report")
log_df = load_log()
report_text = build_report(classes, log_df)
st.code(report_text, language="text")

# Download Buttons
col1, col2 = st.columns(2)
with col1:
    st.download_button("⬇️ Download Report (TXT)", data=report_text, file_name="waste_report.txt", mime="text/plain")
with col2:
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "rb") as f:
            st.download_button("⬇️ Download Full Log (CSV)", data=f, file_name="predictions_log.csv", mime="text/csv")

# Extra Controls
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("📋 Show Recent Predictions"):
        st.dataframe(log_df.sort_values("timestamp", ascending=False).head(200))
with col2:
    if st.button("🗑️ Reset Log"):
        if os.path.exists(LOG_PATH):
            os.remove(LOG_PATH)
        st.rerun()

# Footer
st.markdown('<div class="footer">Developed with ❤️ using Streamlit & TensorFlow</div>', unsafe_allow_html=True)
