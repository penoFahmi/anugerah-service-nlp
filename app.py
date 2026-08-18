# Modul ini mengimplementasikan antarmuka pemrograman aplikasi (API) berbasis FastAPI untuk klasifikasi niat (intent classification).
# Sistem memadukan model semantik FastText untuk ekstraksi fitur teks dan Support Vector Machine (SVM) untuk klasifikasi vektor.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import fasttext
import pickle
import os
import re
import numpy as np
import asyncio

# Struktur data dictionary global dideklarasikan untuk menyimpan instansiasi model ke dalam memori aplikasi.
ml_models = {}

# Skema kelas Pydantic ini mendefinisikan struktur muatan (payload) JSON yang wajib dipenuhi oleh permintaan klien.
class IntentRequest(BaseModel):
    text: str

# Fungsi ini mengeksekusi prapemrosesan teks melalui normalisasi karakter menjadi huruf kecil dan penghapusan tanda baca non-alfanumerik.
# Validasi tipe data diterapkan untuk mengembalikan string kosong apabila masukan bukan bertipe string.
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Fungsi ini mengekstraksi representasi vektor kalimat (sentence vector) melalui penghitungan nilai rata-rata (mean pooling).
# Matriks bernilai nol dikembalikan jika teks masukan tidak memiliki perbendaharaan kata yang valid.
def get_sentence_vector(text, model):
    words = text.split()
    if not words:
        return np.zeros(model.get_dimension())
    
    word_vectors = [model.get_word_vector(w) for w in words]
    return np.mean(word_vectors, axis=0)

# Fungsi sinkron ini merangkum proses pemuatan berkas model biner (Pickle dan FastText) dari sistem berkas lokal.
# Pemisahan fungsi diterapkan untuk memungkinkan eksekusi di dalam utas independen (thread).
def load_models_sync():
    svm_clf = None
    ft_model = None
    
    if os.path.exists('models/svm_model_final.pkl'):
        with open('models/svm_model_final.pkl', 'rb') as f:
            svm_clf = pickle.load(f)
            
    if os.path.exists('models/cc.id.300.bin'):
        ft_model = fasttext.load_model('models/cc.id.300.bin')
        
    return svm_clf, ft_model

# Manajer konteks siklus hidup mengelola tahapan inisialisasi dan penghentian sumber daya aplikasi.
# Pemuatan model didelegasikan ke utas terpisah (asyncio.to_thread) guna mencegah pemblokiran putaran kejadian (event loop).
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tahap inisialisasi (Startup)
    print("Memulai server FastAPI dan memuat model mesin pembelajaran...")
    svm_clf, ft_model = await asyncio.to_thread(load_models_sync)
    ml_models["svm"] = svm_clf
    ml_models["fasttext"] = ft_model
    
    yield
    
    # Tahap penghentian (Shutdown)
    print("Menghentikan server dan membersihkan alokasi memori...")
    ml_models.clear()

# Objek aplikasi FastAPI diinisialisasi dengan menyertakan manajer konteks siklus hidup.
app = FastAPI(
    lifespan=lifespan, 
    title="Anugerah Service NLP", 
    description="Intent Classification menggunakan FastText + SVM"
)

# Titik akhir HTTP POST memproses parameter teks untuk menghasilkan probabilitas kelas niat.
# Validasi ketersediaan model di memori memicu kode status HTTP 500 jika inisialisasi gagal.
@app.post("/api/predict-intent")
async def predict_intent(request: IntentRequest):
    svm_clf = ml_models.get("svm")
    ft_model = ml_models.get("fasttext")
    
    if ft_model is None or svm_clf is None:
        raise HTTPException(status_code=500, detail="Model belum dimuat. Periksa ketersediaan berkas model pada sistem lokal.")
        
    original_text = request.text
    cleaned_text = clean_text(original_text)
    
    # Penugasan niat "lainnya" dieksekusi secara otomatis dengan tingkat keyakinan 0 jika teks masukan kosong setelah prapemrosesan.
    if not cleaned_text:
        return {"intent": "lainnya", "confidence": 0.0, "original_text": original_text}
        
    # Ekstraksi matriks vektor kalimat berbasis FastText dijalankan pada teks yang telah dinormalisasi.
    vector = get_sentence_vector(cleaned_text, ft_model)
    
    # Prediksi menggunakan pengklasifikasi SVM dengan metode evaluasi probabilitas (Platt scaling).
    # Fungsi predict_proba menghasilkan larik dua dimensi yang merepresentasikan distribusi probabilitas pada seluruh kelas.
    probabilities = svm_clf.predict_proba([vector])[0]
    
    # Identifikasi probabilitas tertinggi dilakukan melalui komputasi argmax untuk mendapatkan indeks numerik kelas.
    # Indeks tersebut dipetakan kembali ke label kelas aktual pada atribut kelas pengklasifikasi.
    max_prob_index = np.argmax(probabilities)
    predicted_intent = svm_clf.classes_[max_prob_index]
    confidence_score = float(probabilities[max_prob_index])
    
    print(f"--> [Prediksi] Teks: '{original_text}' | Niat: '{predicted_intent}' | Skor: {confidence_score:.2f}")
    
    return {
        "intent": predicted_intent,
        "confidence": confidence_score,
        "original_text": original_text,
        "cleaned_text": cleaned_text
    }

# Titik akhir HTTP GET yang ditugaskan sebagai layanan pemeriksaan kesehatan (health check) sistem komunikasi internal.
@app.get("/")
def root():
    return {"status": "ok", "message": "Anugerah NLP API is running."}