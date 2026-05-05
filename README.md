# 🧠 MindCheck — Student Depression Screening System
> Sistem Skrining Kesehatan Mental Mahasiswa berbasis Machine Learning
> dengan Klasifikasi Risiko Multilevel

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://checkyourmind.streamlit.app)

---

## 📌 Tentang Proyek

MindCheck adalah sistem skrining kesehatan mental berbasis data
yang dirancang untuk membantu mahasiswa mendeteksi risiko depresi
secara dini. Sistem ini menggunakan model **Logistic Regression**
yang dioptimasi dengan **Optuna**, dan menghasilkan klasifikasi
risiko ke dalam tiga tingkatan berdasarkan ambang batas
probabilitas yang ditentukan secara empiris melalui analisis
Precision-Recall Curve.

Prediksi dilakukan berdasarkan **faktor risiko** seperti tekanan
akademik, stres finansial, jam belajar, dan kebiasaan sehari-hari
— bukan berdasarkan gejala klinis.

> ⚠️ **Disclaimer:** Hasil skrining ini bukan merupakan diagnosis
> medis atau psikologis. Selalu konsultasikan kondisi Anda dengan
> profesional kesehatan mental yang kompeten.

---

## 🚀 Demo Aplikasi

Aplikasi tersedia secara publik dan dapat diakses langsung tanpa
instalasi:

**👉 [checkyourmind.streamlit.app](https://checkyourmind.streamlit.app)**

---

## 🎯 Klasifikasi Risiko

| Level | Probabilitas | Basis Pemilihan |
|---|---|---|
| 🟢 Rendah | P < 33.65% | Recall ≥ 0.95 |
| 🟡 Sedang | 33.65% ≤ P < 70.33% | Zona transisi |
| 🔴 Tinggi | P ≥ 70.33% | Precision ≥ 0.90 |

---

## 📊 Performa Model

| Metrik | Nilai |
|---|---|
| Recall | 0.8356 |
| F1-Score | 0.8209 |
| ROC-AUC | 0.8780 |
| Accuracy | 0.8003 |

> Model dipilih berdasarkan Recall sebagai prioritas utama —
> dalam konteks deteksi depresi, meminimalkan kasus yang tidak
> terdeteksi (*false negative*) lebih penting dari metrik lainnya.

---

## 🔍 Fitur Utama Aplikasi

- **Form 3 langkah** — input terstruktur: data diri, akademik,
  dan gaya hidup
- **Probability gauge** — visualisasi risiko berbentuk setengah
  lingkaran interaktif
- **Rekomendasi adaptif** — saran yang disesuaikan per level
  risiko
- **Hotline support** — kontak layanan kesehatan mental per negara
- **Multi-bahasa** — tersedia dalam Bahasa Indonesia, English,
  dan हिन्दी

---

## 🛠️ Tech Stack

| Komponen | Teknologi |
|---|---|
| Frontend | Streamlit |
| Model | Logistic Regression (scikit-learn) |
| Encoding | Target Encoder (category-encoders) |
| Tuning | Optuna (TPE Sampler, 100 trials) |
| Deployment | Streamlit Community Cloud |

---

## 📂 Dataset

**Student Depression Dataset** · Kaggle — adilshamim8

> Dataset dikumpulkan dari mahasiswa di India. Model mungkin
> paling akurat untuk populasi dengan karakteristik serupa.

🔗 [Akses Dataset](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset)

---

## 🏃 Menjalankan Secara Lokal

```bash
# Clone repository
git clone https://github.com/Kingarthuria/mindcheck-app.git
cd mindcheck-app

# Install dependensi
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

Buka browser di `http://localhost:8501`

---

## 👤 Developer

**Arthur Pendragon**
Data Scientist · Machine Learning Enthusiast

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/arthurpendragon)
[![Email](https://img.shields.io/badge/Email-Contact-red)](mailto:thedumbestknightever@gmail.com)

---

*MindCheck — Sebuah upaya kecil untuk memberi makna
di antara baris kode dan data.*
