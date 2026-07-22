from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import fasttext
import pickle
import os
import re
import numpy as np

app = FastAPI(title="Anugerah Service NLP", description="Intent Classification menggunakan FastText + SVM")

# Global variables for models
ft_model = None
svm_clf = None

class IntentRequest(BaseModel):
    text: str

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def get_sentence_vector(text, model):
    words = text.split()
    if not words:
        return np.zeros(model.get_dimension())
    
    word_vectors = [model.get_word_vector(w) for w in words]
    return np.mean(word_vectors, axis=0)

@app.on_event("startup")
async def startup_event():
    global ft_model, svm_clf
    print("Memulai server FastAPI...")
    
    # 1. Load SVM Model
    if os.path.exists('svm_model.pkl'):
        print("Memuat SVM Model...")
        with open('svm_model.pkl', 'rb') as f:
            svm_clf = pickle.load(f)
    else:
        print("WARNING: svm_model.pkl tidak ditemukan. Model prediksi belum tersedia.")
        
    # 2. Load FastText (cc.id.300.bin)
    if os.path.exists('cc.id.300.bin'):
        print("Memuat FastText cc.id.300.bin (Tunggu beberapa detik)...")
        ft_model = fasttext.load_model('cc.id.300.bin')
        print("FastText berhasil dimuat!")
    else:
        print("WARNING: cc.id.300.bin tidak ditemukan!")

@app.post("/api/predict-intent")
async def predict_intent(request: IntentRequest):
    if ft_model is None or svm_clf is None:
        raise HTTPException(status_code=500, detail="Model belum dimuat di server. Jalankan 2_train_model.py terlebih dahulu.")
        
    original_text = request.text
    cleaned_text = clean_text(original_text)
    
    if not cleaned_text:
        return {"intent": "lainnya", "confidence": 0.0, "original_text": original_text}
        
    # Ekstrak Vektor
    vector = get_sentence_vector(cleaned_text, ft_model)
    
    # Prediksi menggunakan SVM
    # svm_clf.predict_proba mengembalikan array 2D dengan probabilitas tiap kelas
    # svm_clf.classes_ berisi urutan nama intent
    probabilities = svm_clf.predict_proba([vector])[0]
    
    # Cari nilai probability tertinggi
    max_prob_index = np.argmax(probabilities)
    predicted_intent = svm_clf.classes_[max_prob_index]
    confidence_score = float(probabilities[max_prob_index])
    
    # === Log ke Terminal ===
    print(f"--> [Prediksi] Teks: '{original_text}' | Niat: '{predicted_intent}' | Skor: {confidence_score:.2f}")
    
    return {
        "intent": predicted_intent,
        "confidence": confidence_score,
        "original_text": original_text,
        "cleaned_text": cleaned_text
    }

@app.get("/")
def root():
    return {"status": "ok", "message": "Anugerah NLP API is running."}
