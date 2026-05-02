# 🧠 MindCheck — Student Depression Screening System
> Sistem Skrining Depresi Mahasiswa · Multilevel Risk Classification

---

## 📁 Struktur Proyek / Project Structure

```
mindcheck-app/
├── app.py                   ← Aplikasi utama Streamlit
├── requirements.txt         ← Dependensi Python
├── .streamlit/
│   └── config.toml          ← Konfigurasi tema & server
├── models/
│   ├── model.pkl            ← ⚠️  WAJIB: Logistic Regression model
│   ├── scaler.pkl           ← ⚠️  WAJIB: StandardScaler
│   └── target_encoder.pkl   ← ⚠️  WAJIB: Target Encoder (City)
└── README.md
```

---

## ⚡ Ambang Probabilitas / Probability Thresholds

| Level Risiko | Range Probabilitas | Basis |
|---|---|---|
| 🟢 **Rendah / Low**   | P < 0.233  | Recall depresi ≥ 0.95 |
| 🟡 **Sedang / Medium** | 0.233 ≤ P < 0.75 | — |
| 🔴 **Tinggi / High**  | P ≥ 0.75   | Precision depresi ≥ 0.90 |

---

## 🚀 Cara Deploy ke Streamlit Community Cloud

### Langkah 1 — Persiapkan Repository GitHub

```bash
# 1. Buat repo baru di GitHub (nama bebas, misal: mindcheck-app)
# 2. Clone ke lokal
git clone https://github.com/<username>/mindcheck-app.git
cd mindcheck-app
```

### Langkah 2 — Salin Semua File

Salin seluruh isi folder ini ke dalam repo GitHub Anda:
```
app.py
requirements.txt
.streamlit/config.toml
models/model.pkl          ← ⚠️ jangan lupa file ini!
models/scaler.pkl
models/target_encoder.pkl
```

### Langkah 3 — Commit & Push

```bash
git add .
git commit -m "feat: initial MindCheck deployment"
git push origin main
```

### Langkah 4 — Deploy di Streamlit Community Cloud

1. Buka **https://share.streamlit.io** dan login dengan akun GitHub
2. Klik **"New app"**
3. Isi form:
   - **Repository:** `<username>/mindcheck-app`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Klik **"Deploy!"**
5. Tunggu beberapa menit hingga aplikasi live ✅

> 💡 **URL** aplikasi Anda akan berformat:
> `https://<username>-mindcheck-app-app-<hash>.streamlit.app`

---

## 🔧 Konfigurasi Kolom (Penting!)

Pastikan nama kolom di `FEATURE_COLS` dalam `app.py` **persis sama** dengan
nama kolom saat training model:

```python
# app.py — baris ~55
FEATURE_COLS = [
    'Gender',
    'Age',
    'City',
    'Academic Pressure',
    'CGPA',
    'Study Satisfaction',
    'Sleep Duration',
    'Dietary Habits',
    'Education Level',          # ← ubah jika berbeda (misal: 'Degree')
    'Work/Study Hours',
    'Financial Stress',
    'Family History of Mental Illness',
]
```

---

## 🧪 Uji Lokal / Local Testing

```bash
# Install dependensi
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

Buka browser di `http://localhost:8501`

---

## 📦 Dependensi Utama

| Package | Versi Minimum | Kegunaan |
|---|---|---|
| streamlit | 1.32.0 | Framework UI |
| scikit-learn | 1.3.0 | Model & Scaler |
| pandas | 2.0.0 | Preprocessing |
| numpy | 1.24.0 | Komputasi numerik |
| category-encoders | 2.6.0 | Target Encoding |

---

## ⚠️  Catatan Penting / Important Notes

- **File `.pkl` harus ikut di-commit** ke GitHub karena Streamlit Community Cloud
  membaca langsung dari repo. File PKL mahasiswa biasanya kecil (<50 MB) — ini aman.
- Jika PKL Anda **>25 MB**: pertimbangkan Git LFS atau upload ke external storage.
- Aplikasi ini adalah **alat skrining awal**, bukan diagnosis medis.
- Selalu cantumkan disclaimer ketika digunakan secara publik.

---

## 🏗️  Arsitektur Alur Prediksi

```
Input User (Form 3 Langkah)
        ↓
Preprocessing:
  • Clip Age  (atas=35)
  • Clip CGPA (bawah=5.0)
  • Target Encode (City)  ← target_encoder.pkl
  • Select FEATURE_COLS (urutan training)
  • StandardScaler        ← scaler.pkl
        ↓
Logistic Regression     ← model.pkl
        ↓
predict_proba()[1]  →  P(Depression=1)
        ↓
Klasifikasi Risiko:
  P < 0.233  → Rendah / Low
  P < 0.750  → Sedang / Medium
  P ≥ 0.750  → Tinggi / High
        ↓
Tampilkan Hasil + Rekomendasi
```

---

*MindCheck — Dikembangkan sebagai proyek Data Science untuk skrining kesehatan mental mahasiswa.*
