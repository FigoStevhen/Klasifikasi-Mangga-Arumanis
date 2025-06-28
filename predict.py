import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

MODEL_PATH = 'model_lite_V6.keras' 
IMG_SIZE = (224, 224) 

CLASS_NAMES = ['matang', 'muda', 'sangat_matang']

if not os.path.exists(MODEL_PATH):
    print(f"Error: File model tidak ditemukan di '{MODEL_PATH}'")
    model = None
else:
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model berhasil dimuat dari '{MODEL_PATH}'")
        print(f"Urutan kelas yang diharapkan oleh model: {CLASS_NAMES}")
    except Exception as e:
        print(f"Terjadi kesalahan saat memuat model: {e}")
        model = None


def predict_single_image(image_path):
    """
    Fungsi untuk memprediksi kelas dari satu file gambar.

    Args:
        image_path (str): Path lengkap ke file gambar.

    Returns:
        tuple: (str, float) yang berisi (nama_kelas_prediksi, skor_keyakinan)
               atau (None, None) jika terjadi error.
    """
    if model is None:
        print("Model tidak tersedia untuk prediksi.")
        return None, None

    if not os.path.exists(image_path):
        print(f"Error: File gambar tidak ditemukan di '{image_path}'")
        return None, None

    try:
        img = load_img(image_path, target_size=IMG_SIZE)

        img_array = img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_batch)

        prediction_probabilities = model.predict(img_preprocessed)

        score = np.max(prediction_probabilities[0])
        predicted_class_index = np.argmax(prediction_probabilities[0])
        predicted_class_name = CLASS_NAMES[predicted_class_index]

        readable_class_name = predicted_class_name.replace('_', ' ').title()

        return readable_class_name, float(score * 100)

    except Exception as e:
        print(f"Terjadi kesalahan saat memproses gambar: {e}")
        return None, None


if __name__ == '__main__':
    test_image_path = "path/ke/gambar/mangga_uji.jpg" 

    print("\n--- Menjalankan Uji Coba Prediksi ---")
    predicted_class, confidence = predict_single_image(test_image_path)

    if predicted_class:
        print(f"Gambar              : {os.path.basename(test_image_path)}")
        print(f"Hasil Prediksi      : {predicted_class}")
        print(f"Tingkat Keyakinan   : {confidence:.2f}%")
    else:
        print("Gagal melakukan prediksi. Harap periksa path file atau pesan error di atas.")