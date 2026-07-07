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
    logger.info(f"[DITERIMA DARI LARAVEL] Teks asli: '{request.text}'")
    
    # HEURISTIC BYPASS: Jika teks pelanggan sangat pendek dan jelas, langsung beri nilai 1.0
    text_lower = request.text.strip().lower()
    if text_lower in ['setuju', 'oke', 'ok', 'lanjut', 'kerjakan', 'gass']:
        response_data = {"intent": "setuju", "confidence": 1.0, "clean_text": text_lower}
        logger.info(f"[BYPASS HEURISTIC] Hasil Prediksi Mutlak: {response_data}")
        return response_data
        
    if text_lower in ['batal', 'cancel', 'gak jadi', 'ga jadi', 'kemahalan']:
        response_data = {"intent": "batal", "confidence": 1.0, "clean_text": text_lower}
        logger.info(f"[BYPASS HEURISTIC] Hasil Prediksi Mutlak: {response_data}")
        return response_data

    # Jika bukan kata kunci mutlak, jalankan algoritma AI asli (FastText + SVM)
    clean_text = preprocess_text(request.text)
    vector = np.array([ft_model.get_sentence_vector(clean_text)])
    
    prediction = svm_model.predict(vector)[0]
    probabilities = svm_model.predict_proba(vector)[0]
    max_prob = max(probabilities)
    
    response_data = {
        "intent": prediction,
        "confidence": round(float(max_prob), 2),
        "clean_text": clean_text
    }
    
    logger.info(f"[DIKIRIM KE LARAVEL] Hasil Prediksi AI: {response_data}")
    return response_data