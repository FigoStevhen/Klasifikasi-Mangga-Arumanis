# app.py

# Step 1: Import semua library yang dibutuhkan
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
# <<< DIUBAH: Tambahkan import untuk preprocessing EfficientNet >>>
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- Konfigurasi Aplikasi ---
MODEL_PATH = 'model_lite_V6.keras'
IMG_SIZE = (224, 224)

# --- Fungsi-Fungsi Inti ---

# Step 2: Muat model dan ambil nama kelas (dengan caching)
@st.cache_resource
def load_model_and_classes():
    """
    Memuat model Keras dan mendefinisikan nama kelas secara manual.
    Fungsi ini menggunakan cache Streamlit agar model tidak perlu dimuat ulang.
    """
    try:
        model = tf.keras.models.load_model(MODEL_PATH)

        # <<< DIUBAH DAN DIPERBAIKI >>>
        # Urutan harus sesuai dengan abjad folder: 'matang', 'muda', 'sangat_matang'
        class_names = ['matang', 'muda', 'sangat_matang']
        
        return model, class_names
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        st.error(f"Pastikan file '{MODEL_PATH}' berada di lokasi yang benar.")
        return None, None

# Step 3: Fungsi preprocessing dasar
def preprocess_image_from_path(img_path):
    """
    Membaca dan memproses gambar dari path agar sesuai dengan input model.
    """
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    # Tidak perlu cast ke float32 di sini, akan dilakukan oleh preprocess_input
    return tf.expand_dims(img, axis=0)

# --- Tampilan Utama Aplikasi Streamlit ---

st.set_page_config(page_title="Analisis Kematangan Mangga", page_icon="🥭", layout="wide")

st.title("🥭 Aplikasi Analisis Kematangan Mangga")
st.write("Unggah gambar buah mangga untuk diprediksi tingkat kematangannya.")
st.markdown("---")

model, class_names = load_model_and_classes()

if model and class_names:
    # Buat nama kelas yang lebih rapi untuk ditampilkan
    readable_class_names = [name.replace('_', ' ').title() for name in class_names]
    st.success(f"✅ Model **{os.path.basename(MODEL_PATH)}** berhasil dimuat. Kelas yang terdeteksi: **{', '.join(readable_class_names)}**")

    col1, col2 = st.columns([0.8, 1])

    with col1:
        st.header("1. Unggah Gambar Anda")
        uploaded_file = st.file_uploader("Pilih file gambar mangga...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diunggah.")

    with col2:
        st.header("2. Hasil Analisis")
        if uploaded_file is not None:
            # Simpan file sementara untuk diproses
            temp_dir = "temp"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Step 4: Lakukan Prediksi
            with st.spinner("🤖 AI sedang menganalisis gambar..."):
                # Preprocessing dasar
                img_tensor = preprocess_image_from_path(temp_path)
                
                # <<< DIUBAH: Tambahkan preprocessing khusus EfficientNet >>>
                # Ini adalah langkah krusial yang tidak boleh dilewatkan
                img_preprocessed = preprocess_input(img_tensor)

                # Prediksi menggunakan gambar yang sudah sepenuhnya diproses
                pred_probs = model.predict(img_preprocessed)[0]
                
                pred_idx = np.argmax(pred_probs)
                pred_label = class_names[pred_idx]
                confidence_score = np.max(pred_probs)

            # Ubah label menjadi format yang rapi
            readable_pred_label = pred_label.replace('_', ' ').title()
            st.success(f"**Hasil Prediksi: {readable_pred_label}**")
            st.metric(label="Tingkat Keyakinan", value=f"{confidence_score:.2%}")

            # Step 5: Tampilkan distribusi probabilitas
            st.write("#### Distribusi Probabilitas")
            
            # Gunakan nama kelas yang rapi untuk plot
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(readable_class_names, pred_probs, color='orange', alpha=0.7)
            ax.set_ylabel("Probabilitas")
            ax.set_title("Probabilitas Prediksi per Kelas")
            ax.set_ylim([0, 1.1]) # Beri sedikit ruang di atas
            
            # Highlight bar dengan probabilitas tertinggi
            bars[pred_idx].set_color('green')
            
            for i, v in enumerate(pred_probs):
                ax.text(i, v + 0.02, f"{v:.2%}", ha='center', fontweight='bold')

            st.pyplot(fig)

            if os.path.exists(temp_path):
                os.remove(temp_path)
        else:
            st.info("Menunggu gambar diunggah untuk dianalisis.")
else:
    st.warning("Aplikasi tidak dapat berjalan karena model tidak dapat dimuat.")

st.markdown("---")
st.write("Dibuat dengan Streamlit dan TensorFlow")