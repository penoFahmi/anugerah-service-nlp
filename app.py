# from fastapi import FastAPI
# from pydantic import BaseModel
# import fasttext
# import pickle
# import re
# import numpy as np

# # Inisialisasi Aplikasi API
# app = FastAPI(title="Anugerah NLP Service")

# # Muat model ke dalam memori saat server pertama kali menyala
# print("Memuat model FastText dan SVM...")
# ft_model = fasttext.load_model("fasttext_model.bin")
# with open("svm_model.pkl", "rb") as f:
#     svm_model = pickle.load(f)
# print("Model siap digunakan!")

# # Skema data yang diterima API (JSON Payload)
# class TextRequest(BaseModel):
#     text: str

# # Fungsi pembersihan teks (sama dengan yang di Jupyter Notebook)
# def clean_text(text):
#     text = str(text).lower()
#     text = re.sub(r'【.*?】', '', text)
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# # Endpoint utama untuk klasifikasi pesan
# @app.post("/predict")
# def predict_intent(request: TextRequest):
#     cleaned = clean_text(request.text)
    
#     # Jika pesan kosong setelah dibersihkan
#     if not cleaned:
#         return {"label": "Tidak Jelas", "confidence": 0.0, "is_fallback_llm": True}
        
#     # Ubah teks menjadi vektor angka dengan FastText
#     vector = ft_model.get_sentence_vector(cleaned).reshape(1, -1)
    
#     # Prediksi menggunakan SVM dan ambil persentase probabilitasnya
#     probabilities = svm_model.predict_proba(vector)[0]
    
#     # Indeks 0 = Batal, Indeks 1 = Setuju (sesuai urutan training)
#     prob_batal = probabilities[0]
#     prob_setuju = probabilities[1]
    
#     # Tentukan pemenang dan keyakinannya
#     if prob_setuju > prob_batal:
#         confidence = prob_setuju
#         label = "Setuju"
#     else:
#         confidence = prob_batal
#         label = "Batal"
        
#     # Logika ambang batas: lempar ke LLM jika akurasi NLP di bawah 80% (0.80)
#     is_fallback = True if confidence < 0.80 else False
    
#     return {
#         "original_text": request.text,
#         "cleaned_text": cleaned,
#         "label": label,
#         "confidence": float(confidence),
#         "is_fallback_llm": is_fallback
#     }