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

st.set_page_config(page_title="Waste Classifier", layout="centered")

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
    arr = preprocess_input(arr)  # Proper ResNet50 preprocessing
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

# UI
st.title("♻️ Waste Classifier")
st.write("Upload images and get predictions. A persistent log is kept so you can download the report anytime.")

# Load model
try:
    model, classes = load_model()
    target_size = infer_target_size(model)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Uploading images
uploaded = st.file_uploader("📂 Upload one or multiple images", type=["png","jpg","jpeg"], accept_multiple_files=True)

if uploaded:
    if st.button("🚀 Predict and Log"):
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
        st.success("✅ Predictions done and logged.")

        # 🎨 Show results with styling
        for name, label, err in results:
            if err:
                st.error(f"❌ {name} → ERROR: {err}")
            else:
                st.markdown(
                    f"""
                    <div style="padding:15px; margin:10px 0; border-radius:12px; 
                                background-color:#e6ffe6; border:2px solid #2ecc71;">
                        <h4 style="color:#27ae60; margin:0;">📌 {name}</h4>
                        <p style="font-size:22px; font-weight:bold; color:#145a32; margin:5px 0;">
                            Prediction → {label}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# Show current report
st.markdown("---")
st.header("📊 Current Waste Classification Report")
log_df = load_log()
report_text = build_report(classes, log_df)
st.code(report_text, language="text")

# Download buttons
st.download_button("⬇️ Download report (TXT)", data=report_text, file_name="waste_classification_report.txt", mime="text/plain")
if os.path.exists(LOG_PATH):
    with open(LOG_PATH, "rb") as f:
        st.download_button("⬇️ Download full prediction log (CSV)", data=f, file_name="predictions_log.csv", mime="text/csv")

# Small controls
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("📋 Show raw log table"):
        st.dataframe(log_df.sort_values("timestamp", ascending=False).head(200))
with col2:
    if st.button("🗑️ Reset/Delete log"):
        if os.path.exists(LOG_PATH):
            os.remove(LOG_PATH)
        st.rerun()  # ✅ modern Streamlit

st.caption("ℹ️ Notes: The app appends a row for every prediction to Document.csv. This file is used to compute the report.")
