import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import pickle
from PIL import Image

# ==============================
# KONFIGURASI TAMPILAN APP
# ==============================
st.set_page_config(
    page_title="Prediksi Gizi Bungkil Inti Sawit",
    layout="wide",
    page_icon="🧪"
)

# ==============================
# CUSTOM CSS
# ==============================
st.markdown("""
<style>
    .title-text {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #1f2937;
    }
    .subtitle-text {
        text-align: center;
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 25px;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        margin-bottom: 18px;
    }
    .result-card {
        background-color: white;
        padding: 14px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD CNN MODELS
# ==============================
@st.cache_resource
def load_cnn_models():
    model_A = tf.keras.models.load_model("salinansalinanensemble_model_A_mae=0.2405_mse=0.2971.h5")
    model_B = tf.keras.models.load_model("salinansalinanensemble_model_B_mae=0.6032_mse=0.7509.h5")
    model_C = tf.keras.models.load_model("salinansalinanensemble_model_C_mae=0.5556_mse=0.6772.h5")
    return model_A, model_B, model_C

model_A, model_B, model_C = load_cnn_models()

# ==============================
# LOAD MODEL SVR STACKING
# ==============================
@st.cache_resource
def load_svr():
    svr_model = joblib.load("ABC_SVR.pkl")
    scaler = joblib.load("standarscaler.pkl")
    return svr_model, scaler

svr_model, scaler = load_svr()


# ==============================
# PREPROCESS GAMBAR
# ==============================
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((64, 64))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ==============================
# KLASIFIKASI MUTU
# ==============================
def classify_quality(pred, toleransi_protein=1, toleransi_serat=1):
    air = float(pred[1])
    abu = float(pred[2])
    protein = float(pred[3])
    lemak = float(pred[4])
    serat = float(pred[5])
    cangkang = float(pred[6])

    kondisi_mutu1 = (
        air <= 12 and
        abu <= 5 and
        protein >= (16 - toleransi_protein) and
        lemak <= 9 and
        serat <= (16 + toleransi_serat) and
        cangkang <= 10
    )

    if kondisi_mutu1:
        return "Mutu 1"

    kondisi_mutu2 = (
        air <= 12 and
        abu <= 6 and
        protein >= 14 and
        lemak <= 10 and
        serat <= 20 and
        cangkang <= 15
    )

    if kondisi_mutu2:
        return "Mutu 2"

    return "Tidak Mutu"

# ==============================
# UI TAMPILAN
# ==============================
st.markdown("<p class='title-text'>🔬 Prediksi Gizi Bungkil Inti Sawit</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Model Multimodal: CNN Feature Extractor + SVR Stacking</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

# ==============================
# UPLOAD GAMBAR
# ==============================
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("### 📤 Upload Gambar Sampel")

    uploaded = st.file_uploader("Pilih gambar (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Gambar Sampel", width=330)

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# HASIL PREDIKSI
# ==============================
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("### 📊 Hasil Prediksi")

    if uploaded and st.button("🔎 Jalankan Prediksi", use_container_width=True):
        try:
            img_array = preprocess_image(img)

            # CNN FEATURE
            feat_A = model_A.predict(img_array)[0]
            feat_B = model_B.predict(img_array)[0]
            feat_C = model_C.predict(img_array)[0]

            # GABUNGKAN FITUR
            X_ABC = np.concatenate([feat_A, feat_B, feat_C]).reshape(1, -1)

            # PREDIKSI SVR
            y_scaled = svr_model.predict(X_ABC)
            y_pred = scaler.inverse_transform(y_scaled.reshape(1, -1))[0]

            labels = [
                "Bahan Kering","Air","Abu","Protein Kasar",
                "Lemak Kasar","Serat Kasar","Cangkang"
            ]

            for i, lab in enumerate(labels):
                st.markdown(
                    f"<div class='result-card'><b>{lab}:</b> {y_pred[i]:.2f}</div>",
                    unsafe_allow_html=True
                )

            kategori = classify_quality(y_pred)
            st.success(f"🏷️ Klasifikasi Mutu: **{kategori}**")

        except Exception as e:
            st.error(f"❌ Error saat prediksi: {e}")

    st.markdown("</div>", unsafe_allow_html=True)







