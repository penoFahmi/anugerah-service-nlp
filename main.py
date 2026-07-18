import logging
from fastapi import FastAPI
from pydantic import BaseModel
from gensim.models import FastText
import joblib
import re
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import warnings

warnings.filterwarnings('ignore')

# 1. Konfigurasi Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 2. Inisialisasi aplikasi FastAPI
app = FastAPI(title="NLP Intent Recognition API (Gensim FastText + SVM)")

# 3. Muat model yang sudah dilatih (pastikan file ada di folder yang sama)
logger.info("Memuat model FastText dan SVM...")
try:
    ft_model = FastText.load("fasttext_model.gensim")
    svm_model = joblib.load("svm_model.joblib")
    logger.info("Model berhasil dimuat!")
except Exception as e:
    logger.error(f"Gagal memuat model: {e}")

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

# Fungsi ekstrak fitur kalimat dari Gensim FastText
def get_sentence_vector(text: str, model):
    words = text.split()
    vectors = [model.wv[w] for w in words if w in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# 6. Buat endpoint untuk menerima data POST
@app.post("/api/predict-intent")
def predict_intent(request: MessageRequest):
    logger.info(f"[DITERIMA DARI LARAVEL] Teks asli: '{request.text}'")
    


    # Proses AI (FastText + SVM)
    clean_text = preprocess_text(request.text)
    vector = get_sentence_vector(clean_text, ft_model).reshape(1, -1)
    
    prediction = svm_model.predict(vector)[0]
    probabilities = svm_model.predict_proba(vector)[0]
    max_prob = max(probabilities)
    
    response_data = {
        "intent": str(prediction),
        "confidence": round(float(max_prob), 2),
        "clean_text": clean_text
    }
    
    logger.info(f"[DIKIRIM KE LARAVEL] Hasil Prediksi AI: {response_data}")
    return response_data