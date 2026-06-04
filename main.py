from fastapi import FastAPI
from pydantic import BaseModel
import fasttext
import pickle
import re
import numpy as np

app = FastAPI(title="NLP Engine Anugerah Computer")

# 1. LOAD MODEL YANG SUDAH DITRAINING
print("⏳ Memuat model FastText dan SVM...")
ft_model = fasttext.load_model("fasttext_model.bin")
with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

# 2. DEFINISIKAN AMBANG BATAS (THRESHOLD)
THRESHOLD = 0.75

class MessageRequest(BaseModel):
    message: str

def clean_text(text):
    text = str(text).lower()
    return re.sub(r'[^a-z0-9\s]', '', text).strip()

@app.post("/predict")
def predict_intent(req: MessageRequest):
    # 1. Bersihkan teks
    teks_bersih = clean_text(req.message)
    
    # 2. Ekstraksi Vektor dengan FastText
    vektor = ft_model.get_sentence_vector(teks_bersih)
    vektor_reshaped = np.array(vektor).reshape(1, -1)
    
    # 3. Klasifikasi SVM dan Ambil Probabilitas (Platt Scaling)
    pred_kelas = svm_model.predict(vektor_reshaped)[0]
    probabilitas = svm_model.predict_proba(vektor_reshaped)[0]
    skor_tertinggi = max(probabilitas)
    
    # 4. Logika Ambang Batas (Sesuai Proposal Skripsi)
    if skor_tertinggi < THRESHOLD or pred_kelas == "UMUM_TEKNIS":
        status_akhir = "FALLBACK_LLM"
    else:
        status_akhir = pred_kelas

    return {
        "pesan_asli": req.message,
        "intent_svm": pred_kelas,
        "skor_probabilitas": float(skor_tertinggi),
        "keputusan_sistem": status_akhir
    }

# Endpoint untuk cek server hidup
@app.get("/")
def read_root():
    return {"status": "NLP Engine Berjalan Normal"}