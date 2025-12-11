import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image
from keras.layers import InputLayer

# ==============================
# KONFIGURASI TAMPILAN APP
# ==============================
st.set_page_config(
    page_title="Prediksi Gizi Bungkil Inti Sawit",
    layout="wide",
    page_icon="🧪"
)

# Custom CSS untuk dashboard elegan
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }
    .title-text {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #1f2937;
        margin-bottom: -10px;
    }
    .subtitle-text {
        text-align: center;
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 30px;
    }
    .card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.07);
        margin-bottom: 20px;
    }
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.10);
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL CNN (FEATURE EXTRACTOR)
# ==============================
@st.cache_resource
def load_cnn_models():
    model_A = tf.keras.models.load_model(
        "salinansalinanensemble_model_A_mae=0.2405_mse=0.2971.h5",
        custom_objects={"InputLayer": InputLayer}
    )
    model_B = tf.keras.models.load_model(
        "salinansalinanensemble_model_B_mae=0.6032_mse=0.7509.h5",
        custom_objects={"InputLayer": InputLayer}
    )
    model_C = tf.keras.models.load_model(
        "salinansalinanensemble_model_C_mae=0.5556_mse=0.6772.h5",
        custom_objects={"InputLayer": InputLayer}
    )
    return model_A, model_B, model_C

model_A, model_B, model_C = load_cnn_models()

# ==============================
# LOAD MODEL STACKING XGB
# ==============================
@st.cache_resource
def load_stacking_model():
    svr_model = joblib.load("ABC_SVR.pkl")
    scaler = joblib.load("standarscaler.pkl")
    return svr_model, scaler

svr_model, scaler = load_stacking_model()

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
    # pred = array dengan urutan:
    # [Bahan Kering, Air, Abu, Protein, Lemak, Serat, Cangkang]

    air = float(pred[1])
    abu = float(pred[2])
    protein = float(pred[3])
    lemak = float(pred[4])
    serat = float(pred[5])
    cangkang = float(pred[6])

    # =============================
    # LOGIKA MUTU 1 + BORDERLINE
    # =============================
    kondisi_mutu1 = (
        air <= 12 and
        abu <= 5 and
        protein >= 16 - toleransi_protein and
        lemak <= 9 and
        serat <= 16 + toleransi_serat and
        cangkang <= 10
    )

    if kondisi_mutu1:
        # Borderline jika protein <16 ATAU serat >16
        if protein < 16 or serat > 16:
            return "Mutu 1"
        else:
            return "Mutu 1"

    # =============================
    # LOGIKA MUTU 2
    # =============================
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

    # =============================
    # TIDAK MUTU
    # =============================
    return "Tidak Mutu"


# ==============================
# UI DASHBOARD
# ==============================

# Title
st.markdown("<p class='title-text'>🔬 Prediksi Gizi Bungkil Inti Sawit</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Model Multimodal: CNN Feature Extractor + SVR Stacking</p>", unsafe_allow_html=True)

# Layout utama
col_left, col_right = st.columns([1, 2])

# ==============================
# BOX UPLOAD GAMBAR
# ==============================
with col_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("### 📤 Upload Gambar Sampel")
    uploaded = st.file_uploader("Pilih file gambar (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Gambar Sampel", width=350)

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# BOX HASIL PREDIKSI
# ==============================
with col_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("### 📊 Hasil Prediksi")

    if uploaded and st.button("🔎 Jalankan Prediksi", use_container_width=True):
        try:
            img_array = preprocess_image(image)

            # Ekstraksi fitur CNN
            feat_A = model_A.predict(img_array)[0]
            feat_B = model_B.predict(img_array)[0]
            feat_C = model_C.predict(img_array)[0]

            # Concatenate feature (berdasarkan training)
            X_ABC = np.concatenate([feat_A, feat_B, feat_C]).reshape(1, -1)

            # Prediksi
            y_scaled = svr_model.predict(X_ABC)
            y_pred = scaler.inverse_transform(y_scaled)[0]

            target_names = [
                "Bahan Kering","Air","Abu","Protein Kasar",
                "Lemak Kasar","Serat Kasar","Cangkang"
            ]

            # Tampilkan prediksi dalam card
            for i, name in enumerate(target_names):
                st.markdown(
                    f"<div class='result-card'><b>{name}:</b> {y_pred[i]:.2f}</div>",
                    unsafe_allow_html=True
                )

            # Klasifikasi mutu
            kategori = classify_quality(y_pred)
            st.success(f"🏷️ Klasifikasi Mutu: **{kategori}**")

        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
