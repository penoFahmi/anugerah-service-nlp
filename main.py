import logging
from fastapi import FastAPI
from pydantic import BaseModel
import fasttext
import joblib
import re
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# 1. Konfigurasi Logging
# Mengatur format log agar menampilkan waktu, tingkat pesan (INFO), dan isi pesannya
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 2. Inisialisasi aplikasi FastAPI
app = FastAPI(title="NLP Intent Recognition API")

# 3. Muat model yang sudah dilatih (pastikan file ada di folder yang sama)
logger.info("Memuat model FastText dan SVM...")
ft_model = fasttext.load_model("fasttext_model.bin")
svm_model = joblib.load("svm_model.pkl")
logger.info("Model berhasil dimuat!")

# 4. Siapkan pembersih teks
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# 5. Tentukan struktur data yang akan diterima dari Laravel
class MessageRequest(BaseModel):
    text: str

# Fungsi untuk membersihkan teks
def preprocess_text(text: str):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return stemmer.stem(text)

# 6. Buat endpoint untuk menerima data POST
@app.post("/api/predict-intent")
def predict_intent(request: MessageRequest):
    # LOG: Mencatat data yang diterima dari Laravel
    logger.info(f"[DITERIMA DARI LARAVEL] Teks asli: '{request.text}'")
    
    # Bersihkan teks yang masuk
    clean_text = preprocess_text(request.text)
    
    # Ubah teks menjadi vektor matematika
    vector = np.array([ft_model.get_sentence_vector(clean_text)])
    
    # Lakukan prediksi kelas dan ambil nilai keyakinannya (probabilitas)
    prediction = svm_model.predict(vector)[0]
    probabilities = svm_model.predict_proba(vector)[0]
    max_prob = max(probabilities)
    
    # Siapkan data yang akan dikembalikan
    response_data = {
        "intent": prediction,
        "confidence": round(float(max_prob), 2),
        "clean_text": clean_text
    }
    
    # LOG: Mencatat data yang akan dilempar kembali ke Laravel
    logger.info(f"[DIKIRIM KE LARAVEL] Hasil Prediksi: {response_data}")
    
    # Kembalikan hasil dalam bentuk JSON
    return response_data