# Anugerah NLP Service - Intent Classification AI

Microservice kecerdasan buatan berbasis Python yang bertugas sebagai mesin utama klasifikasi niat (Intent Classification) dari percakapan pelanggan via WhatsApp untuk **Anugerah Computer**. 

Sistem ini didesain sebagai bagian dari arsitektur *Hybrid AI*, bekerja berdampingan dengan Backend Laravel dan Fallback LLM (Groq).

## 🧠 Arsitektur Machine Learning

Pipeline Natural Language Processing (NLP) ini menggunakan pendekatan ML konvensional yang ringan namun akurat:
1. **Cleansing & Preprocessing:** Membersihkan teks dari tanda baca berlebih dan *stopwords*.
2. **Stemming (Sastrawi):** Mengembalikan kata ke bentuk dasarnya (khusus Bahasa Indonesia).
3. **Word Embedding (FastText):** Mengubah teks menjadi representasi vektor numerik menggunakan pre-trained model `cc.id.300.bin` (300 Dimensi) dari Facebook/Meta.
4. **Klasifikasi (SVM):** Model Support Vector Machine (SVM) digunakan untuk memprediksi probabilitas intent berdasarkan vektor kata. 

Jika akurasi/probabilitas SVM berada di bawah ambang batas **75%**, sistem secara otomatis menyerahkan (fallback) tugas pemahaman bahasa ke LLM via Laravel.

## 📁 Struktur Repositori

- `main.py` - File utama server FastAPI yang menangani HTTP Webhook dari Laravel.
- `Notebooks/` - Folder berisi Jupyter Notebook (`1_Data_Preparation.ipynb`, `2_Model_Training.ipynb`) untuk eksperimen akademis dan pelatihan ulang (retrain) model secara lokal.
- `models/` - Folder penyimpanan model klasifikasi hasil latih (`svm_model.pkl` dan metadatanya).
- `requirements.txt` - Daftar pustaka/library Python yang dibutuhkan.

## 🚀 Panduan Instalasi (Development)

Sangat disarankan menggunakan **Conda** atau **Virtual Environment (venv)** untuk menghindari konflik library Python.

1. **Clone repositori ini:**
   ```bash
   git clone https://github.com/penoFahmi/anugerah-service-nlp.git
   cd anugerah-service-nlp
   ```

2. **Buat & Aktifkan Virtual Environment (Contoh dengan Conda):**
   ```bash
   conda create -n nlp-anugerah python=3.10
   conda activate nlp-anugerah
   ```

3. **Install Dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Unduh Model FastText:**
   Sistem ini membutuhkan model embedding Bahasa Indonesia dari FastText.
   - Unduh file `cc.id.300.bin.gz` dari [FastText Website](https://fasttext.cc/docs/en/crawl-vectors.html).
   - Ekstrak file tersebut hingga Anda mendapatkan file `cc.id.300.bin`.
   - Letakkan file `cc.id.300.bin` tepat di direktori utama (root) repositori ini.

5. **Jalankan Server API:**
   ```bash
   uvicorn main:app --reload --port 8001
   ```
   *Server akan berjalan di `http://localhost:8001`. Anda dapat membuka `http://localhost:8001/docs` untuk melihat dokumentasi interaktif (Swagger UI).*

## 🔌 API Endpoint

- `POST /webhook`
  Endpoint utama yang dipanggil oleh Laravel.
  **Payload:** `{"message": "Bang, laptop saya Asus ROG tiba-tiba mati total", "phone": "08123456789"}`
  **Response:** `{"intent": "konsultasi_kerusakan", "confidence": 0.82}`

---
*Dikembangkan untuk operasional Anugerah Computer & Keperluan Akademis Skripsi.*