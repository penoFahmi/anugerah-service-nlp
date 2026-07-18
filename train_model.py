import pandas as pd
import re
import numpy as np
from gensim.models import FastText
import joblib
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

print("=== 1. Memulai Proses Data ===")
# 1. Buka data mentah ASLI
df_raw = pd.read_csv('master_chat_wa.csv')

# 2. Filter Chat Admin (Nomor Anugerah Computer) dan Media
nomor_admin = 6285752887496
df_pelanggan = df_raw[df_raw['Phone Number'] != nomor_admin].copy()
df_pelanggan = df_pelanggan.dropna(subset=['Message Body'])
df_pelanggan = df_pelanggan[~df_pelanggan['Message Body'].astype(str).str.contains('【IMAGE】|【VIDEO】|【STICKER】|【DOCUMENT】')]

# 3. Setup Stemmer
print("Inisialisasi Stemmer Sastrawi...")
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def bersihkan_teks(teks):
    teks = str(teks).lower()
    # Hapus karakter non-alfabet
    teks = re.sub(r'[^a-z\s]', '', teks) 
    teks = re.sub(r'\s+', ' ', teks).strip()
    # Stemming
    return stemmer.stem(teks)

print("Membersihkan teks dan melakukan stemming (Proses ini mungkin memakan waktu)...")
df_pelanggan['Teks_Bersih'] = df_pelanggan['Message Body'].apply(bersihkan_teks)
# Hapus pesan yang kosong setelah dibersihkan
df_pelanggan = df_pelanggan[df_pelanggan['Teks_Bersih'].str.len() > 0]

# 4. Fungsi Auto-Labeling (Keyword Matching)
def auto_label(teks):
    teks = str(teks).lower()
    if any(kata in teks for kata in ['status', 'sampai mana', 'kapan', 'progress', 'kelar', 'selesai', 'belum', 'udah jadi', 'siap']):
        return 'tanya_status'
    elif any(kata in teks for kata in ['batal', 'cancel', 'tidak jadi', 'ndak usah', 'tarik', 'kemahalan', 'gak jadi', 'ga jadi', 'mahal']):
        return 'batal'
    elif any(kata in teks for kata in ['setuju', 'oke', 'lanjut', 'gas', 'deal', 'sikat', 'kerja', 'ok', 'sip', 'ya', 'boleh']):
        return 'setuju'
    elif any(kata in teks for kata in ['garansi', 'ram', 'biaya', 'harga', 'nego', 'jam', 'buka', 'lcd', 'ganti', 'berapa', 'ongkos', 'tutup']):
        return 'umum_teknis'
    else:
        return 'TIDAK_DIKETAHUI'

# Gunakan teks asli (huruf kecil) untuk label agar lebih akurat sebelum distem
df_pelanggan['Label'] = df_pelanggan['Message Body'].apply(auto_label)

# Ambil hanya yang terdeteksi labelnya
df_final = df_pelanggan[df_pelanggan['Label'] != 'TIDAK_DIKETAHUI'].copy()

# Pastikan setiap kelas memiliki minimal 5 sampel (karena cv=5 pada GridSearch)
# Menggunakan teknik Oversampling sederhana (karena ini skripsi, agar model AI murni yang belajar label 'batal')
label_counts = df_final['Label'].value_counts()
print(f"Distribusi Kelas Awal:\n{label_counts}")

for label, count in label_counts.items():
    if count < 5:
        minority_data = df_final[df_final['Label'] == label]
        duplicates_needed = 5 - count
        oversampled = minority_data.sample(n=duplicates_needed, replace=True, random_state=42)
        df_final = pd.concat([df_final, oversampled])

print(f"Distribusi Kelas Setelah Oversampling:\n{df_final['Label'].value_counts()}")
print(f"Total data bersih siap latih: {len(df_final)} baris.")

print("\n=== 2. Pelatihan Word Embedding (FastText Gensim) ===")
sentences = [text.split() for text in df_final['Teks_Bersih']]

print("Melatih model FastText (dim=100)...")
ft_model = FastText(sentences=sentences, vector_size=100, window=3, min_count=1, epochs=50)

# Fungsi untuk Sentence Embedding
def get_sentence_vector(text, model):
    words = text.split()
    vectors = [model.wv[w] for w in words if w in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

print("Ekstraksi Vektor Fitur...")
X = np.array([get_sentence_vector(text, ft_model) for text in df_final['Teks_Bersih']])
y = df_final['Label']


print("\n=== 3. Pelatihan Model Klasifikasi (SVM) ===")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Mencari parameter SVM terbaik (Grid Search)...")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

svm_base = SVC(probability=True, random_state=42)
grid_search = GridSearchCV(estimator=svm_base, param_grid=param_grid, cv=5, verbose=0, scoring='f1_macro')
grid_search.fit(X_train, y_train)

best_svm_model = grid_search.best_estimator_
print(f"Parameter SVM Terbaik: {grid_search.best_params_}")

print("\n=== 4. Evaluasi Model ===")
y_pred = best_svm_model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\n=== 5. Menyimpan Model ===")
ft_model.save("fasttext_model.gensim")
joblib.dump(best_svm_model, 'svm_model.joblib')
print("Model berhasil disimpan: 'fasttext_model.gensim' dan 'svm_model.joblib'")
print("SELESAI!")
