import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import zipfile
import os

# ==================================================
# Extract Model from ZIP (if needed)
# ==================================================
if not os.path.exists("iot_intrusion_ann.keras"):
    if os.path.exists("model.zip"):
        with zipfile.ZipFile("model.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        st.success("📦 Model extracted successfully!")

# ==================================================
# Load Model & Scaler
# ==================================================
model = load_model("iot_intrusion_ann.keras")
scaler = joblib.load("scaler.pkl")

# ==================================================
# Page Configuration
# ==================================================
st.set_page_config(
    page_title="IoT Intrusion Detection using ANN",
    page_icon="🛡️",
    layout="wide"
)

# ==================================================
# Sidebar
# ==================================================
st.sidebar.title("🛡️ About Project")

st.sidebar.markdown("""
## IoT Intrusion Detection using ANN

This application detects whether network traffic is:

- ✅ Normal  
- 🚨 Intrusion  

using a Deep Learning (ANN) model.

---

### Model Performance
- 🎯 Accuracy : 98.16%
- 🎯 Precision : 99.17%
- 🎯 Recall : 98.33%
- 🎯 F1-Score : 98.75%
- 🎯 ROC-AUC : 99.70%

---

### Technologies
- TensorFlow / Keras  
- Streamlit  
- Scikit-learn  
- Pandas / NumPy  
""")

# ==================================================
# Main Title
# ==================================================
st.title("🛡️ IoT Intrusion Detection using ANN")

st.info(
    "Upload a CSV file containing IoT network traffic features "
    "to classify records as Normal or Intrusion."
)

# ==================================================
# Expected Features
# ==================================================
features = [
    'Src Port', 'Dst Port', 'Protocol', 'Flow Duration',
    'Total Fwd Packet', 'Total Bwd packets',
    'Total Length of Fwd Packet', 'Fwd Packet Length Min',
    'Bwd Packet Length Max', 'Flow IAT Min', 'Fwd IAT Min',
    'Fwd Header Length', 'Bwd Packets/s', 'Packet Length Max',
    'Packet Length Std', 'RST Flag Count', 'FWD Init Win Bytes',
    'Bwd Init Win Bytes', 'Idle Mean', 'Idle Max',
    'FIN Flag Count', 'SYN Flag Count', 'ACK Flag Count'
]

# ==================================================
# Upload File
# ==================================================
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

# ==================================================
# Prediction Section
# ==================================================
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Raw Data")
    st.dataframe(df.head(), use_container_width=True)

    # Remove Label if exists
    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    # Missing values
    if df.isnull().sum().sum() > 0:
        st.warning("⚠ Missing values detected → filling with 0")
        df = df.fillna(0)

    # Column validation
    missing_cols = set(features) - set(df.columns)
    extra_cols = set(df.columns) - set(features)

    if missing_cols or extra_cols:
        st.error(f"""
❌ Column mismatch detected!

Missing columns: {missing_cols}
Extra columns: {extra_cols}
""")
        st.stop()

    df = df[features]

    # ==================================================
    # Dataset Overview
    # ==================================================
    st.subheader("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("📄 Rows", df.shape[0])
    col2.metric("📊 Features", df.shape[1])
    col3.metric("🧮 Total Cells", df.shape[0] * df.shape[1])

    # ==================================================
    # Scaling + Prediction
    # ==================================================
    status = st.empty()
    progress = st.progress(0)

    for i in range(100):
        if i == 30:
            status.text("🔄 Scaling data...")
            X = scaler.transform(df)

        if i == 70:
            status.text("🧠 Running ANN model...")
            probabilities = model.predict(X, verbose=0)

        progress.progress(i + 1)

    # ==================================================
    # Results
    # ==================================================
    predictions = (probabilities > 0.5).astype(int)

    result = df.copy()
    result["Prediction"] = np.where(
        predictions.flatten() == 1,
        "Intrusion",
        "Normal"
    )

    result["Probability"] = probabilities.flatten()

    intrusion_count = (result["Prediction"] == "Intrusion").sum()
    normal_count = (result["Prediction"] == "Normal").sum()

    st.success("🎯 Prediction Completed Successfully!")

    c1, c2, c3 = st.columns(3)
    c1.metric("🚨 Intrusions", intrusion_count)
    c2.metric("✅ Normal", normal_count)
    c3.metric("📊 Total", len(result))

    # ==================================================
    # Pie Chart
    # ==================================================
    st.subheader("📊 Prediction Distribution")

    fig, ax = plt.subplots()

    ax.pie(
        [normal_count, intrusion_count],
        labels=["Normal", "Intrusion"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#2ecc71", "#e74c3c"],
        explode=(0.05, 0.05)
    )

    ax.axis("equal")
    st.pyplot(fig)

    # ==================================================
    # Styled Table
    # ==================================================
    st.subheader("📋 Results")

    def style_pred(val):
        return (
            "background-color:#e74c3c;color:white;font-weight:bold"
            if val == "Intrusion"
            else "background-color:#2ecc71;color:white;font-weight:bold"
        )

    st.dataframe(
        result.style.applymap(style_pred, subset=["Prediction"]),
        use_container_width=True
    )

    # ==================================================
    # Download
    # ==================================================
    csv = result.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Results",
        data=csv,
        file_name="iot_intrusion_predictions.csv",
        mime="text/csv"
    )

    # Reset
    if st.button("🔄 Analyze Another File"):
        st.rerun()

# ==================================================
# Footer
# ==================================================
st.markdown("---")

st.markdown("""
<center>
<h4>🛡️ IoT Intrusion Detection using ANN</h4>
<p>Developed by <b>Mai</b> | AI & Data Science Project</p>
</center>
""", unsafe_allow_html=True)