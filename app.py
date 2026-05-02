"""
MindCheck — Sistem Skrining Depresi Mahasiswa
           Student Depression Screening System

Multi-level Risk Classification via Probability Thresholds:
  Low Risk    : P(depression) < 0.3365
  Medium Risk : 0.3365 ≤ P(depression) < 0.703
  High Risk   : P(depression) ≥ 0.703
"""

import math
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MindCheck | Student Depression Screening",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

THRESHOLD_LOW  = 0.3365
THRESHOLD_HIGH = 0.703

LANG_ID = "🇮🇩  Bahasa Indonesia"
LANG_EN = "🇬🇧  English"
LANG_HI = "🇮🇳  हिन्दी"
LANG_OPTIONS = [LANG_ID, LANG_EN, LANG_HI]

CITIES = sorted([
    'Agra', 'Ahmedabad', 'Bangalore', 'Bhopal', 'Chennai', 'Delhi',
    'Faridabad', 'Ghaziabad', 'Hyderabad', 'Indore', 'Jaipur', 'Kalyan',
    'Kanpur', 'Kolkata', 'Lucknow', 'Ludhiana', 'Meerut', 'Mumbai',
    'Nagpur', 'Nashik', 'Patna', 'Pune', 'Rajkot', 'Srinagar', 'Surat',
    'Thane', 'Vadodara', 'Varanasi', 'Vasai-Virar', 'Visakhapatnam',
])
CITIES.append('Others')

FEATURE_COLS = [
    'Gender', 'Age', 'City', 'Academic Pressure', 'CGPA',
    'Study Satisfaction', 'Sleep Duration', 'Dietary Habits',
    'Education Level', 'Work/Study Hours', 'Financial Stress',
    'Family History of Mental Illness',
]

# ══════════════════════════════════════════════════════════════════════════════
#  TRANSLATIONS
# ══════════════════════════════════════════════════════════════════════════════

STRINGS = {
    LANG_ID: {
        # Header
        "tagline": "Sistem Skrining Kesehatan Mental Mahasiswa",
        # Sidebar
        "nav_screening":   "  Skrining",
        "nav_project":     "  Tentang Proyek",
        "nav_developer":   "  Tentang Pembuat",
        "not_medical":     "Bukan diagnosis medis",
        "language":        "Bahasa / Language",
        # Step labels
        "step1_label": "Data Diri",
        "step2_label": "Akademik",
        "step3_label": "Gaya Hidup",
        # Step 1
        "step1_title": "Data Diri",
        "step1_sub":   "Mohon isi informasi dasar berikut ini",
        "gender":      "Jenis Kelamin",
        "male":        "Laki-laki",
        "female":      "Perempuan",
        "edu_level":   "Jenjang Pendidikan",
        "bachelor":    "Bachelor / S1",
        "master":      "Master / S2",
        "doctorate":   "Doktor / S3",
        "age":         "Usia (tahun)",
        "age_help":    "Usia di atas 35 tahun akan diproses sebagai 35 oleh model",
        "city":        "Kota",
        "city_help":   "Pilih 'Others' jika kota Anda tidak ada dalam daftar",
        # Step 2
        "step2_title": "Kehidupan Akademik",
        "step2_sub":   "Informasi terkait pengalaman akademik Anda",
        "acad_press":  "Tekanan Akademik",
        "acad_help":   "1 = Sangat rendah   ·   5 = Sangat tinggi",
        "cgpa":        "IPK (skala 1–10)",
        "cgpa_help":   "Nilai di bawah 5.0 akan diproses sebagai 5.0 oleh model",
        "study_sat":   "Kepuasan Belajar",
        "study_help":  "1 = Sangat tidak puas   ·   5 = Sangat puas",
        "study_hours": "Jam Belajar per Hari",
        "study_hours_caption": "{v} jam per hari",
        # Step 3
        "step3_title": "Gaya Hidup & Kesehatan",
        "step3_sub":   "Informasi tentang kebiasaan harian dan kondisi Anda",
        "sleep":       "Durasi Tidur Harian",
        "sleep_lt5":   "Kurang dari 5 jam",
        "sleep_56":    "5–6 jam",
        "sleep_78":    "7–8 jam",
        "sleep_gt8":   "Lebih dari 8 jam",
        "diet":        "Kebiasaan Makan",
        "diet_unhealthy": "Tidak Sehat",
        "diet_moderate":  "Cukup",
        "diet_healthy":   "Sehat",
        "fin_stress":  "Stres Finansial",
        "fin_help":    "1 = Tidak ada   ·   5 = Sangat tinggi",
        "fam_history": "Riwayat Keluarga dengan Gangguan Mental",
        "fam_help":    "Apakah ada anggota keluarga inti yang pernah didiagnosis gangguan mental?",
        "no":          "Tidak",
        "yes":         "Ya",
        # Buttons
        "btn_next":    "Lanjut →",
        "btn_back":    "← Kembali",
        "btn_result":  "Lihat Hasil",
        "btn_restart": "Isi Ulang",
        "btn_retry":   "← Ulangi",
        # Result
        "prob_label":  "Probabilitas Depresi",
        "rec_header":  "💡 Rekomendasi",
        # Disclaimer
        "disclaimer": (
            "⚠️ <strong>Penting:</strong> "
            "Hasil skrining ini <em>bukan merupakan diagnosis medis atau psikologis</em>. "
            "Sistem ini dikembangkan untuk tujuan edukasi dan skrining awal berbasis data "
            "menggunakan model Logistic Regression dengan klasifikasi risiko multilevel. "
            "Untuk penilaian yang akurat, selalu konsultasikan kondisi Anda dengan "
            "profesional kesehatan mental yang kompeten."
        ),
        # About Project
        "about_project_title": "Tentang Proyek",
        "about_project_sub":   "Pengembangan Sistem Skrining Depresi Mahasiswa",
        # About Developer
        "about_dev_title": "Developer",
        "about_dev_sub":   "Orang di balik MindCheck",
        # Error
        "err_processing": "❌ **Terjadi kesalahan saat memproses data:**",
        "err_file":       "❌ **File model tidak ditemukan.**",
    },

    LANG_EN: {
        # Header
        "tagline": "Student Mental Health Screening System",
        # Sidebar
        "nav_screening":   "  Screening",
        "nav_project":     "  About Project",
        "nav_developer":   "  About Developer",
        "not_medical":     "Not a medical diagnosis",
        "language":        "Language / Bahasa",
        # Step labels
        "step1_label": "Personal",
        "step2_label": "Academic",
        "step3_label": "Lifestyle",
        # Step 1
        "step1_title": "Personal Information",
        "step1_sub":   "Please fill in your basic information",
        "gender":      "Gender",
        "male":        "Male",
        "female":      "Female",
        "edu_level":   "Education Level",
        "bachelor":    "Bachelor",
        "master":      "Master",
        "doctorate":   "Doctorate",
        "age":         "Age (years)",
        "age_help":    "Ages above 35 will be processed as 35 by the model",
        "city":        "City",
        "city_help":   "Select 'Others' if your city is not listed",
        # Step 2
        "step2_title": "Academic Life",
        "step2_sub":   "Information related to your academic experience",
        "acad_press":  "Academic Pressure",
        "acad_help":   "1 = Very low   ·   5 = Very high",
        "cgpa":        "CGPA (scale 1–10)",
        "cgpa_help":   "Values below 5.0 will be processed as 5.0 by the model",
        "study_sat":   "Study Satisfaction",
        "study_help":  "1 = Very unsatisfied   ·   5 = Very satisfied",
        "study_hours": "Study Hours per Day",
        "study_hours_caption": "{v} hours per day",
        # Step 3
        "step3_title": "Lifestyle & Wellbeing",
        "step3_sub":   "Information about your daily habits and condition",
        "sleep":       "Daily Sleep Duration",
        "sleep_lt5":   "Less than 5 hours",
        "sleep_56":    "5–6 hours",
        "sleep_78":    "7–8 hours",
        "sleep_gt8":   "More than 8 hours",
        "diet":        "Dietary Habits",
        "diet_unhealthy": "Unhealthy",
        "diet_moderate":  "Moderate",
        "diet_healthy":   "Healthy",
        "fin_stress":  "Financial Stress",
        "fin_help":    "1 = None   ·   5 = Very high",
        "fam_history": "Family History of Mental Illness",
        "fam_help":    "Has any immediate family member been diagnosed with a mental disorder?",
        "no":          "No",
        "yes":         "Yes",
        # Buttons
        "btn_next":    "Next →",
        "btn_back":    "← Back",
        "btn_result":  "See Result",
        "btn_restart": "Restart",
        "btn_retry":   "← Retry",
        # Result
        "prob_label":  "Depression Probability",
        "rec_header":  "💡 Recommendations",
        # Disclaimer
        "disclaimer": (
            "⚠️ <strong>Important:</strong> "
            "This screening result is <em>not a medical or psychological diagnosis</em>. "
            "This system was developed for educational purposes and preliminary data-based "
            "screening using a Logistic Regression model with multilevel risk classification. "
            "For an accurate assessment, always consult a qualified mental health professional."
        ),
        # About Project
        "about_project_title": "About Project",
        "about_project_sub":   "Student Depression Screening System Development",
        # About Developer
        "about_dev_title": "Developer",
        "about_dev_sub":   "The person behind MindCheck",
        # Error
        "err_processing": "❌ **Error processing data:**",
        "err_file":       "❌ **Model files not found.**",
    },

    LANG_HI: {
        # Header
        "tagline": "छात्र मानसिक स्वास्थ्य जांच प्रणाली",
        # Sidebar
        "nav_screening":   "  जांच",
        "nav_project":     "  परियोजना के बारे में",
        "nav_developer":   "  डेवलपर के बारे में",
        "not_medical":     "यह चिकित्सीय निदान नहीं है",
        "language":        "भाषा / Language",
        # Step labels
        "step1_label": "व्यक्तिगत",
        "step2_label": "शैक्षणिक",
        "step3_label": "जीवनशैली",
        # Step 1
        "step1_title": "व्यक्तिगत जानकारी",
        "step1_sub":   "कृपया नीचे दी गई बुनियादी जानकारी भरें",
        "gender":      "लिंग",
        "male":        "पुरुष",
        "female":      "महिला",
        "edu_level":   "शिक्षा स्तर",
        "bachelor":    "स्नातक / Bachelor",
        "master":      "स्नातकोत्तर / Master",
        "doctorate":   "डॉक्टरेट / Doctorate",
        "age":         "आयु (वर्ष)",
        "age_help":    "35 वर्ष से अधिक आयु को मॉडल द्वारा 35 के रूप में संसाधित किया जाएगा",
        "city":        "शहर",
        "city_help":   "यदि आपका शहर सूची में नहीं है तो 'Others' चुनें",
        # Step 2
        "step2_title": "शैक्षणिक जीवन",
        "step2_sub":   "आपके शैक्षणिक अनुभव से संबंधित जानकारी",
        "acad_press":  "शैक्षणिक दबाव",
        "acad_help":   "1 = बहुत कम   ·   5 = बहुत अधिक",
        "cgpa":        "सीजीपीए (पैमाना 1–10)",
        "cgpa_help":   "5.0 से कम मान को मॉडल द्वारा 5.0 के रूप में संसाधित किया जाएगा",
        "study_sat":   "अध्ययन संतुष्टि",
        "study_help":  "1 = बिल्कुल असंतुष्ट   ·   5 = बहुत संतुष्ट",
        "study_hours": "प्रतिदिन अध्ययन के घंटे",
        "study_hours_caption": "प्रतिदिन {v} घंटे",
        # Step 3
        "step3_title": "जीवनशैली और स्वास्थ्य",
        "step3_sub":   "आपकी दैनिक आदतों और स्थिति के बारे में जानकारी",
        "sleep":       "दैनिक नींद की अवधि",
        "sleep_lt5":   "5 घंटे से कम",
        "sleep_56":    "5–6 घंटे",
        "sleep_78":    "7–8 घंटे",
        "sleep_gt8":   "8 घंटे से अधिक",
        "diet":        "आहार की आदतें",
        "diet_unhealthy": "अस्वस्थ",
        "diet_moderate":  "सामान्य",
        "diet_healthy":   "स्वस्थ",
        "fin_stress":  "वित्तीय तनाव",
        "fin_help":    "1 = बिल्कुल नहीं   ·   5 = बहुत अधिक",
        "fam_history": "मानसिक बीमारी का पारिवारिक इतिहास",
        "fam_help":    "क्या किसी निकट परिवार के सदस्य को मानसिक विकार का निदान हुआ है?",
        "no":          "नहीं",
        "yes":         "हाँ",
        # Buttons
        "btn_next":    "आगे →",
        "btn_back":    "← वापस",
        "btn_result":  "परिणाम देखें",
        "btn_restart": "फिर से शुरू करें",
        "btn_retry":   "← पुनः प्रयास",
        # Result
        "prob_label":  "अवसाद की संभावना",
        "rec_header":  "💡 सुझाव",
        # Disclaimer
        "disclaimer": (
            "⚠️ <strong>महत्वपूर्ण:</strong> "
            "यह जांच परिणाम <em>चिकित्सीय या मनोवैज्ञानिक निदान नहीं है</em>। "
            "यह प्रणाली शैक्षणिक उद्देश्यों के लिए विकसित की गई है। "
            "सटीक मूल्यांकन के लिए हमेशा किसी योग्य मानसिक स्वास्थ्य विशेषज्ञ से परामर्श लें।"
        ),
        # About Project
        "about_project_title": "परियोजना के बारे में",
        "about_project_sub":   "छात्र अवसाद जांच प्रणाली का विकास",
        # About Developer
        "about_dev_title": "डेवलपर",
        "about_dev_sub":   "MindCheck के पीछे का व्यक्ति",
        # Error
        "err_processing": "❌ **डेटा संसाधित करने में त्रुटि:**",
        "err_file":       "❌ **मॉडल फ़ाइलें नहीं मिलीं।**",
    },
}


def s(key: str) -> str:
    """Return translated string for current language."""
    lang = st.session_state.get("lang", LANG_ID)
    return STRINGS[lang].get(key, key)


# ══════════════════════════════════════════════════════════════════════════════
#  RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

RECOMMENDATIONS = {
    "low": {
        LANG_ID: {
            "title": "✅ Risiko Rendah",
            "desc": (
                "Berdasarkan respons Anda, tidak terdapat indikator signifikan yang "
                "mengarah pada risiko depresi saat ini. Pertahankan kondisi positif ini "
                "dan terus jaga kesehatan mental Anda!"
            ),
            "recs": [
                ("🛏️  Jaga Keseimbangan Hidup", [
                    "Pertahankan durasi tidur 7–8 jam dan pola makan seimbang secara konsisten",
                ]),
                ("📓  Pemantauan Diri", [
                    "Lakukan refleksi atau journaling mingguan untuk memantau kondisi mental Anda",
                ]),
                ("🏫  Manfaatkan Konseling Preventif", [
                    "Layanan konseling kampus tersedia untuk semua mahasiswa — tidak harus menunggu hingga bermasalah",
                ]),
            ],
        },
        LANG_EN: {
            "title": "✅ Low Risk",
            "desc": (
                "Based on your responses, there are no significant indicators pointing "
                "to a depression risk at this time. Keep up the great habits and "
                "continue nurturing your mental wellbeing!"
            ),
            "recs": [
                ("🛏️  Maintain Life Balance", [
                    "Maintain 7–8 hours of sleep and a balanced diet consistently",
                ]),
                ("📓  Self-Monitoring", [
                    "Practice weekly reflection or journaling to monitor your mental state",
                ]),
                ("🏫  Use Preventive Counseling", [
                    "Campus counseling services are available for all students — you don't have to wait until things get worse",
                ]),
            ],
        },
        LANG_HI: {
            "title": "\u2705 \u0915\u092e \u091c\u094b\u0916\u093f\u092e",
            "desc": (
                "\u0906\u092a\u0915\u0940 \u092a\u094d\u0930\u0924\u093f\u0915\u094d\u0930\u093f\u092f\u093e\u0913\u0902 \u0915\u0947 \u0906\u0927\u093e\u0930 \u092a\u0930, \u0905\u092d\u0940 \u0905\u0935\u0938\u093e\u0926 \u0915\u0947 \u091c\u094b\u0916\u093f\u092e \u0915\u0940 \u0913\u0930 \u0907\u0936\u093e\u0930\u093e \u0915\u0930\u0928\u0947 \u0935\u093e\u0932\u0947 "
                "\u0915\u094b\u0908 \u092e\u0939\u0924\u094d\u0935\u092a\u0942\u0930\u094d\u0923 \u0938\u0902\u0915\u0947\u0924\u0915 \u0928\u0939\u0940\u0902 \u0939\u0948\u0902\u0964 \u0907\u0938 \u0938\u0915\u093e\u0930\u093e\u0924\u094d\u092e\u0915 \u0938\u094d\u0925\u093f\u0924\u093f \u0915\u094b \u092c\u0928\u093e\u090f \u0930\u0916\u0947\u0902 "
                "\u0914\u0930 \u0905\u092a\u0928\u0947 \u092e\u093e\u0928\u0938\u093f\u0915 \u0938\u094d\u0935\u093e\u0938\u094d\u0925\u094d\u092f \u0915\u093e \u0916\u094d\u092f\u093e\u0932 \u0930\u0916\u0924\u0947 \u0930\u0939\u0947\u0902!"
            ),
            "recs": [
                ("\U0001f6cf\ufe0f  \u091c\u0940\u0935\u0928 \u0938\u0902\u0924\u0941\u0932\u0928 \u092c\u0928\u093e\u090f \u0930\u0916\u0947\u0902", [
                    "\u0932\u0917\u093e\u0924\u093e\u0930 7\u20138 \u0918\u0902\u091f\u0947 \u0915\u0940 \u0928\u0940\u0902\u0926 \u0914\u0930 \u0938\u0902\u0924\u0941\u0932\u093f\u0924 \u0906\u0939\u093e\u0930 \u092c\u0928\u093e\u090f \u0930\u0916\u0947\u0902",
                ]),
                ("\U0001f4d3  \u0938\u094d\u0935-\u0928\u093f\u0917\u0930\u093e\u0928\u0940", [
                    "\u0905\u092a\u0928\u0940 \u092e\u093e\u0928\u0938\u093f\u0915 \u0938\u094d\u0925\u093f\u0924\u093f \u0915\u0940 \u0928\u093f\u0917\u0930\u093e\u0928\u0940 \u0915\u0947 \u0932\u093f\u090f \u0938\u093e\u092a\u094d\u0924\u093e\u0939\u093f\u0915 \u092a\u094d\u0930\u0924\u093f\u092c\u093f\u0902\u092c \u092f\u093e \u091c\u0930\u094d\u0928\u0932\u093f\u0902\u0917 \u0915\u0930\u0947\u0902",
                ]),
                ("\U0001f3eb  \u0928\u093f\u0935\u093e\u0930\u0915 \u092a\u0930\u093e\u092e\u0930\u094d\u0936 \u0915\u093e \u0909\u092a\u092f\u094b\u0917 \u0915\u0930\u0947\u0902", [
                    "\u0915\u0948\u0902\u092a\u0938 \u0915\u093e\u0909\u0902\u0938\u0932\u093f\u0902\u0917 \u0938\u0947\u0935\u093e\u090f\u0902 \u0938\u092d\u0940 \u091b\u093e\u0924\u094d\u0930\u094b\u0902 \u0915\u0947 \u0932\u093f\u090f \u0909\u092a\u0932\u092c\u094d\u0927 \u0939\u0948\u0902 \u2014 \u0938\u092e\u0938\u094d\u092f\u093e \u092c\u0922\u093c\u0928\u0947 \u0915\u093e \u0907\u0902\u0924\u091c\u093e\u0930 \u0928 \u0915\u0930\u0947\u0902",
                ]),
            ],
        },
        "hotline": False,
    },

    "medium": {
        LANG_ID: {
            "title": "⚠️  Risiko Sedang",
            "desc": (
                "Skrining mendeteksi beberapa faktor yang perlu mendapat perhatian. "
                "Kondisi ini belum tentu merupakan depresi, namun penanganan dini "
                "sangat dianjurkan agar tidak berkembang lebih lanjut."
            ),
            "recs": [
                ("🏫  Jadwalkan Konseling", [
                    "Segera jadwalkan sesi dengan konselor atau psikolog kampus Anda — ini langkah paling penting",
                ]),
                ("📖  Kelola Tekanan Akademik", [
                    "Bagi tugas besar menjadi bagian kecil dan komunikasikan kesulitan kepada dosen atau wali studi",
                ]),
                ("😴  Prioritaskan Istirahat", [
                    "Targetkan 7–8 jam tidur konsisten — kualitas tidur berpengaruh langsung pada kesehatan mental",
                ]),
                ("💬  Jangan Hadapi Sendiri", [
                    "Ceritakan kondisi Anda kepada orang yang dipercaya — teman dekat, keluarga, atau konselor",
                    "Jika sewaktu-waktu merasa kewalahan, layanan hotline tersedia untuk diajak bicara",
                ]),
            ],
        },
        LANG_EN: {
            "title": "⚠️  Medium Risk",
            "desc": (
                "The screening detected several factors that need attention. "
                "This does not necessarily indicate depression, but early intervention "
                "is strongly recommended to prevent further progression."
            ),
            "recs": [
                ("🏫  Schedule Counseling", [
                    "Schedule a session with your campus counselor or psychologist soon — this is the most important step",
                ]),
                ("📖  Manage Academic Pressure", [
                    "Break large tasks into smaller parts and communicate challenges to your lecturer or academic advisor",
                ]),
                ("😴  Prioritize Rest", [
                    "Aim for a consistent 7–8 hours of sleep — sleep quality directly affects mental health",
                ]),
                ("💬  Don't Face It Alone", [
                    "Share your condition with someone you trust — a close friend, family member, or counselor",
                    "If you ever feel overwhelmed, hotline services are available to talk to",
                ]),
            ],
        },
        LANG_HI: {
            "title": "\u26a0\ufe0f  \u092e\u0927\u094d\u092f\u092e \u091c\u094b\u0916\u093f\u092e",
            "desc": (
                "\u091c\u093e\u0902\u091a \u092e\u0947\u0902 \u0915\u0941\u091b \u090f\u0938\u0947 \u0915\u093e\u0930\u0915 \u092e\u093f\u0932\u0947 \u0939\u0948\u0902 \u091c\u093f\u0928 \u092a\u0930 \u0927\u094d\u092f\u093e\u0928 \u0926\u0947\u0928\u0947 \u0915\u0940 \u091c\u0930\u0942\u0930\u0924 \u0939\u0948\u0964 "
                "\u092f\u0939 \u091c\u0930\u0942\u0930\u0940 \u0928\u0939\u0940\u0902 \u0915\u093f \u092f\u0939 \u0905\u0935\u0938\u093e\u0926 \u0939\u094b, \u0932\u0947\u0915\u093f\u0928 \u0906\u0917\u0947 \u092c\u0922\u093c\u0928\u0947 \u0938\u0947 \u0930\u094b\u0915\u0928\u0947 \u0915\u0947 \u0932\u093f\u090f "
                "\u0936\u0940\u0918\u094d\u0930 \u0939\u0938\u094d\u0924\u0915\u094d\u0937\u0947\u092a \u0915\u0940 \u0926\u0943\u0922\u093c\u0924\u093e \u0938\u0947 \u0938\u093f\u092b\u093e\u0930\u093f\u0936 \u0915\u0940 \u091c\u093e\u0924\u0940 \u0939\u0948\u0964"
            ),
            "recs": [
                ("\U0001f3eb  \u092a\u0930\u093e\u092e\u0930\u094d\u0936 \u0936\u0947\u0921\u094d\u092f\u0942\u0932 \u0915\u0930\u0947\u0902", [
                    "\u091c\u0932\u094d\u0926 \u0938\u0947 \u091c\u0932\u094d\u0926 \u0905\u092a\u0928\u0947 \u0915\u0948\u0902\u092a\u0938 \u0915\u093e\u0909\u0902\u0938\u0932\u0930 \u092f\u093e \u092e\u0928\u094b\u0935\u0948\u091c\u094d\u091e\u093e\u0928\u093f\u0915 \u0915\u0947 \u0938\u093e\u0925 \u0938\u0924\u094d\u0930 \u0936\u0947\u0921\u094d\u092f\u0942\u0932 \u0915\u0930\u0947\u0902",
                ]),
                ("\U0001f4d6  \u0936\u0948\u0915\u094d\u0937\u0923\u093f\u0915 \u0926\u092c\u093e\u0935 \u0915\u094b \u092a\u094d\u0930\u092c\u0902\u0927\u093f\u0924 \u0915\u0930\u0947\u0902", [
                    "\u092c\u0921\u093c\u0947 \u0915\u093e\u0930\u094d\u092f\u094b\u0902 \u0915\u094b \u091b\u094b\u091f\u0947 \u0939\u093f\u0938\u094d\u0938\u094b\u0902 \u092e\u0947\u0902 \u092c\u093e\u0902\u091f\u0947\u0902 \u0914\u0930 \u0915\u0920\u093f\u0928\u093e\u0907\u092f\u094b\u0902 \u0915\u094b \u0905\u092a\u0928\u0947 \u0936\u093f\u0915\u094d\u0937\u0915 \u092f\u093e \u0938\u0932\u093e\u0939\u0915\u093e\u0930 \u0938\u0947 \u0938\u093e\u091d\u093e \u0915\u0930\u0947\u0902",
                ]),
                ("\U0001f634  \u0906\u0930\u093e\u092e \u0915\u094b \u092a\u094d\u0930\u093e\u0925\u092e\u093f\u0915\u0924\u093e \u0926\u0947\u0902", [
                    "\u0932\u0917\u093e\u0924\u093e\u0930 7\u20138 \u0918\u0902\u091f\u0947 \u0915\u0940 \u0928\u0940\u0902\u0926 \u0915\u093e \u0932\u0915\u094d\u0937\u094d\u092f \u0930\u0916\u0947\u0902 \u2014 \u0928\u0940\u0902\u0926 \u0915\u0940 \u0917\u0941\u0923\u0935\u0924\u094d\u0924\u093e \u0938\u0940\u0927\u0947 \u092e\u093e\u0928\u0938\u093f\u0915 \u0938\u094d\u0935\u093e\u0938\u094d\u0925\u094d\u092f \u0915\u094b \u092a\u094d\u0930\u092d\u093e\u0935\u093f\u0924 \u0915\u0930\u0924\u0940 \u0939\u0948",
                ]),
                ("\U0001f4ac  \u0905\u0915\u0947\u0932\u0947 \u092e\u0924 \u091d\u0947\u0932\u0947\u0902", [
                    "\u0905\u092a\u0928\u0940 \u0938\u094d\u0925\u093f\u0924\u093f \u0915\u093f\u0938\u0940 \u0935\u093f\u0936\u094d\u0935\u0938\u0928\u0940\u092f \u0935\u094d\u092f\u0915\u094d\u0924\u093f \u0915\u0947 \u0938\u093e\u0925 \u0938\u093e\u091d\u093e \u0915\u0930\u0947\u0902",
                    "\u0905\u0917\u0930 \u0915\u092d\u0940 \u0905\u092d\u093f\u092d\u0942\u0924 \u092e\u0939\u0938\u0942\u0938 \u0939\u094b\u0902, \u0924\u094b \u092c\u093e\u0924 \u0915\u0930\u0928\u0947 \u0915\u0947 \u0932\u093f\u090f \u0939\u0949\u091f\u0932\u093e\u0907\u0928 \u0938\u0947\u0935\u093e\u090f\u0902 \u0909\u092a\u0932\u092c\u094d\u0927 \u0939\u0948\u0902",
                ]),
            ],
        },
        "hotline": True,
    },

    "high": {
        LANG_ID: {
            "title": "🚨  Risiko Tinggi",
            "desc": (
                "Hasil skrining menunjukkan indikator yang signifikan terkait risiko depresi. "
                "Ini bukan diagnosis medis, namun Anda sangat disarankan untuk segera "
                "mencari bantuan dari profesional kesehatan mental."
            ),
            "recs": [
                ("🏥  Cari Bantuan Profesional — PRIORITAS UTAMA", [
                    "Hubungi psikolog atau psikiater sesegera mungkin — jangan tunda",
                ]),
                ("💬  Berbagi dengan Orang Terpercaya", [
                    "Ceritakan perasaan Anda kepada keluarga, sahabat dekat, atau mentor",
                    "Anda tidak harus menanggung beban ini sendirian",
                ]),
                ("🎓  Minta Penyesuaian Akademik", [
                    "Bicarakan dengan dosen atau wali studi tentang perpanjangan deadline atau pengurangan beban studi",
                ]),
                ("📵  Jaga Kesehatan Digital", [
                    "Batasi paparan informasi negatif dan istirahatkan diri dari media sosial untuk sementara waktu",
                ]),
                ("📞  Hubungi Hotline Jika Dibutuhkan", [
                    "Hotline tersedia 24 jam — Anda tidak sendiri",
                ]),
            ],
        },
        LANG_EN: {
            "title": "🚨  High Risk",
            "desc": (
                "The screening shows significant indicators of depression risk. "
                "This is not a medical diagnosis, but you are strongly encouraged to "
                "seek help from a mental health professional immediately."
            ),
            "recs": [
                ("🏥  Seek Professional Help — TOP PRIORITY", [
                    "Contact a psychologist or psychiatrist as soon as possible — don't delay",
                ]),
                ("💬  Talk to Someone You Trust", [
                    "Share your feelings with family, close friends, or a trusted mentor",
                    "You don't have to carry this burden alone",
                ]),
                ("🎓  Request Academic Adjustment", [
                    "Discuss deadline extensions or reduced study loads with your lecturer or academic advisor",
                ]),
                ("📵  Protect Your Digital Wellbeing", [
                    "Limit exposure to negative content and take a temporary break from social media",
                ]),
                ("📞  Contact Hotline if Needed", [
                    "Hotlines are available 24 hours — you are not alone",
                ]),
            ],
        },
        "hotline": True,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  HOTLINES
# ══════════════════════════════════════════════════════════════════════════════

HOTLINES = {
    LANG_ID: [
        ("🔴", "Into The Light Indonesia", "119 ext 8"),
        ("🟡", "Yayasan Pulih",            "(021) 788-42580"),
        ("🟢", "Kemenkes Sehat Jiwa",      "1500-454"),
    ],
    LANG_EN: [
        ("🔴", "International Association for Suicide Prevention", "www.iasp.info/resources/Crisis_Centres"),
        ("🟡", "Befrienders Worldwide",    "www.befrienders.org"),
        ("🟢", "Crisis Text Line",         "Text HOME to 741741"),
    ],
    LANG_HI: [
        ("🔴", "iCall",                    "9152987821"),
        ("🟡", "Vandrevala Foundation",    "1860-2662-345"),
        ("🟢", "NIMHANS",                  "080-46110007"),
    ],
}

HOTLINE_STRINGS = {
    "medium": {
        LANG_ID: ("💬 Layanan Berbicara", "Tersedia jika kamu butuh seseorang untuk diajak bicara"),
        LANG_EN: ("💬 Talk Services",     "Available if you need someone to talk to"),
        LANG_HI: ("💬 बातचीत सेवा",       "उपलब्ध है यदि आपको बात करने के लिए किसी की जरूरत है"),
    },
    "high": {
        LANG_ID: ("📞 Hotline Darurat",   "Hubungi sekarang — kamu tidak sendiri"),
        LANG_EN: ("📞 Emergency Hotline", "Call now — you are not alone"),
        LANG_HI: ("📞 आपातकालीन हॉटलाइन", "अभी कॉल करें — आप अकेले नहीं हैं"),
    },
}


def render_hotline(risk: str):
    lang     = st.session_state.get("lang", LANG_ID)
    lines    = HOTLINES.get(lang, HOTLINES[LANG_ID])
    header, sub = HOTLINE_STRINGS[risk][lang]
    bg = "background:linear-gradient(135deg,#1B3A52,#1D6F8E)" if risk == "medium" \
        else "background:linear-gradient(135deg,#7B1B1B,#C0392B)"

    items_html = "".join(
        f'<p style="color:#BADAEA;font-size:.83rem;margin:.25rem 0;">'
        f'{dot} &nbsp;<strong style="color:#fff;">{name}</strong>'
        f' &nbsp;→&nbsp; {number}</p>'
        for dot, name, number in lines
    )
    st.markdown(f"""
    <div style="{bg};border-radius:14px;padding:1.1rem 1.4rem;margin-top:.8rem;">
        <h5 style="color:#fff;font-size:.88rem;margin:0 0 .2rem;">{header}</h5>
        <p style="color:#BADAEA;font-size:.75rem;margin:0 0 .6rem;font-style:italic;">{sub}</p>
        {items_html}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Lora:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    .block-container { padding-top:1.5rem !important; padding-bottom:5rem !important; max-width:740px !important; }
    .mc-header  { text-align:center; padding:2rem 1rem 1.2rem; }
    .mc-title   { font-family:'Lora',serif; font-size:2rem; color:#1B3A52; margin:0 0 .3rem; letter-spacing:-.5px; }
    .mc-tagline { color:#5B7F99; font-size:.88rem; font-weight:400; }
    .step-row   { display:flex; align-items:center; justify-content:center; margin:1.4rem 0 .4rem; }
    .step-circle { width:34px; height:34px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:.82rem; font-weight:700; transition:all .3s; }
    .step-circle.done    { background:#2ECC71; color:#fff; }
    .step-circle.active  { background:#1D6F8E; color:#fff; box-shadow:0 0 0 5px rgba(29,111,142,.18); }
    .step-circle.pending { background:#DCE8F0; color:#8BAABB; }
    .step-line      { height:2px; width:60px; background:#DCE8F0; }
    .step-line.done { background:#2ECC71; }
    .step-label-row { display:flex; align-items:flex-start; justify-content:center; margin-bottom:1.4rem; }
    .step-label     { font-size:.68rem; font-weight:500; text-align:center; width:100px; color:#8BAABB; line-height:1.4; }
    .step-label.active { color:#1D6F8E; }
    .step-label.done   { color:#2ECC71; }
    .step-label-spacer { width:60px; }
    .mc-card { background:#fff; border-radius:18px; padding:1.8rem 2rem; box-shadow:0 2px 24px rgba(0,50,100,.07); margin-bottom:1rem; }
    .mc-card-header { display:flex; align-items:center; gap:.6rem; margin-bottom:.2rem; }
    .mc-card-title  { font-family:'Lora',serif; font-size:1.1rem; color:#1B3A52; margin:0; }
    .mc-card-sub    { color:#7A9EAF; font-size:.82rem; margin-bottom:1.2rem; }
    .mc-divider     { border:none; border-top:1px solid #EEF4F8; margin:1.1rem 0; }
    .stButton > button { border-radius:10px !important; font-family:'DM Sans',sans-serif !important; font-weight:600 !important; letter-spacing:.1px !important; transition:all .2s ease !important; border:none !important; }
    .stButton > button[kind="primary"]   { background:#1D6F8E !important; color:#fff !important; }
    .stButton > button[kind="primary"]:hover { background:#155E79 !important; box-shadow:0 4px 16px rgba(29,111,142,.35) !important; transform:translateY(-1px) !important; }
    .stButton > button[kind="secondary"] { background:#EEF4F8 !important; color:#4A7B9A !important; }
    .stButton > button[kind="secondary"]:hover { background:#DCE8F0 !important; }
    .result-wrap { border-radius:20px; padding:2rem 2rem 1.5rem; text-align:center; margin:1rem 0 1.2rem; }
    .result-wrap.low    { background:linear-gradient(135deg,#D5F5E3,#A9DFBF); border:2px solid #27AE60; }
    .result-wrap.medium { background:linear-gradient(135deg,#FDEBD0,#FAD7A0); border:2px solid #F39C12; }
    .result-wrap.high   { background:linear-gradient(135deg,#FADBD8,#F1948A); border:2px solid #E74C3C; }
    .result-title { font-family:'Lora',serif; font-size:1.65rem; font-weight:700; margin-bottom:.3rem; }
    .result-title.low    { color:#1E8449; }
    .result-title.medium { color:#9A6400; }
    .result-title.high   { color:#C0392B; }
    .result-sub { font-size:.85rem; opacity:.75; margin-bottom:.1rem; }
    .rec-section { margin-top:1rem; }
    .rec-header  { font-family:'Lora',serif; font-size:1.1rem; color:#1B3A52; margin-bottom:.8rem; }
    .rec-card    { background:#F4F9FC; border-left:4px solid #1D6F8E; border-radius:0 10px 10px 0; padding:.9rem 1.2rem; margin-bottom:.7rem; }
    .rec-card h5 { color:#1D5C78; font-size:.88rem; font-weight:600; margin:0 0 .35rem; }
    .rec-card ul { color:#3D6275; font-size:.85rem; margin:0; padding-left:1.2rem; line-height:1.75; }
    .disclaimer  { background:#FFFCE6; border:1px solid #F5D000; border-radius:10px; padding:.9rem 1.1rem; font-size:.79rem; color:#6B5900; margin-top:1.3rem; line-height:1.65; }
    div[data-testid="stSlider"] > label, div[data-testid="stSelectbox"] > label,
    div[data-testid="stNumberInput"] > label, div[data-testid="stRadio"] > label
        { font-weight:500 !important; color:#1B3A52 !important; font-size:.92rem !important; }
    div[data-testid="stSlider"] [role="slider"] { background-color:#1D6F8E !important; }
    div[data-testid="stCaptionContainer"] p { color:#7A9EAF !important; }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#0F2535 0%,#1B3A52 100%) !important; }
    [data-testid="stSidebar"] * { color:#C8DEE9 !important; }
    [data-testid="stSidebar"] .stButton > button { background:rgba(255,255,255,0.08) !important; color:#C8DEE9 !important; border:1px solid rgba(255,255,255,0.12) !important; text-align:center !important; font-weight:400 !important; width:100% !important; padding:.45rem .9rem !important; }
    [data-testid="stSidebar"] .stButton > button:hover { background:rgba(255,255,255,0.16) !important; border-color:rgba(255,255,255,0.25) !important; color:#fff !important; }
    .sb-logo    { text-align:center; padding:1.4rem 0 .6rem; font-size:2rem; }
    .sb-brand   { text-align:center; font-family:'Lora',serif; font-size:1.1rem; color:#fff !important; letter-spacing:-.3px; margin-bottom:.8rem; }
    .sb-section { font-size:.68rem; font-weight:700; letter-spacing:1.5px; color:#5B8FAA !important; text-transform:uppercase; padding:.4rem 1rem .5rem; border-bottom:1px solid rgba(255,255,255,0.08); margin-bottom:.4rem; }
    .sb-divider { border:none; border-top:1px solid #253D50; margin:.6rem .8rem; }
    .about-hero       { background:linear-gradient(135deg,#0F2535,#1D6F8E); border-radius:20px; padding:2rem 2rem 1.6rem; text-align:center; margin-bottom:1.2rem; }
    .about-hero-icon  { font-size:2.4rem; margin-bottom:.4rem; }
    .about-hero-title { font-family:'Lora',serif; font-size:1.7rem; color:#fff; margin:0 0 .3rem; }
    .about-hero-sub   { color:#8DC4DB; font-size:.85rem; }
    .about-card       { background:#fff; border-radius:16px; padding:1.5rem 1.8rem; margin-bottom:1rem; box-shadow:0 2px 18px rgba(0,50,100,.07); }
    .about-card-title { font-family:'Lora',serif; font-size:1.05rem; color:#1B3A52; margin-bottom:.8rem; }
    .about-card p  { color:#3D5E75; font-size:.88rem; line-height:1.8; margin:0 0 .5rem; }
    .about-card ul { color:#3D5E75; font-size:.88rem; line-height:1.9; padding-left:1.2rem; margin:0; }
    .threshold-row { display:flex; gap:.8rem; margin:.5rem 0; flex-wrap:wrap; }
    .threshold-badge { flex:1; min-width:140px; border-radius:12px; padding:.8rem 1rem; text-align:center; }
    .threshold-badge.low    { background:#D5F5E3; border:1.5px solid #27AE60; }
    .threshold-badge.medium { background:#FDEBD0; border:1.5px solid #E67E22; }
    .threshold-badge.high   { background:#FADBD8; border:1.5px solid #E74C3C; }
    .threshold-badge .tb-label { font-size:.72rem; font-weight:600; text-transform:uppercase; letter-spacing:.8px; margin-bottom:.2rem; }
    .threshold-badge.low    .tb-label { color:#1E8449; }
    .threshold-badge.medium .tb-label { color:#9A6400; }
    .threshold-badge.high   .tb-label { color:#C0392B; }
    .threshold-badge .tb-range { font-size:.95rem; font-weight:700; color:#1B3A52; }
    .threshold-badge .tb-basis { font-size:.72rem; color:#6B8EA0; margin-top:.15rem; }
    .tech-pill { display:inline-block; background:#EEF4F8; border-radius:20px; padding:.3rem .85rem; font-size:.8rem; font-weight:500; color:#1D6F8E; margin:.25rem .2rem; }
    .dev-card  { background:linear-gradient(135deg,#F0F6FA,#fff); border-radius:16px; padding:1.5rem 1.8rem; border:1.5px solid #DCE8F0; margin-bottom:1rem; }
    .dev-avatar { width:72px; height:72px; border-radius:50%; background:linear-gradient(135deg,#1D6F8E,#1B3A52); display:flex; align-items:center; justify-content:center; font-size:1.8rem; margin:0 auto .8rem; }
    .dev-name { font-family:'Lora',serif; font-size:1.2rem; color:#1B3A52; margin:0 0 .15rem; text-align:center; }
    .dev-role { font-size:.82rem; color:#5B8FAA; margin:0 0 .8rem; text-align:center; }
    .dev-links a { display:inline-flex; align-items:center; gap:.3rem; font-size:.82rem; color:#1D6F8E; text-decoration:none; margin:.2rem .4rem; font-weight:500; }
    .dev-links a:hover { text-decoration:underline; }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="⏳ Loading model…")
def load_models():
    base = Path(__file__).parent / "models"
    with open(base / "model.pkl",          "rb") as f: model  = pickle.load(f)
    with open(base / "scaler.pkl",         "rb") as f: scaler = pickle.load(f)
    with open(base / "target_encoder.pkl", "rb") as f: te     = pickle.load(f)
    return model, scaler, te


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING & PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(raw: dict, scaler, te) -> np.ndarray:
    df = pd.DataFrame([raw])
    df['Age']  = df['Age'].clip(upper=35)
    df['CGPA'] = df['CGPA'].clip(lower=5.0)
    for col in FEATURE_COLS:
        if col != 'City':
            df[col] = df[col].astype(float)
    X         = df[FEATURE_COLS].copy()
    X_encoded = te.transform(X)
    return scaler.transform(X_encoded[FEATURE_COLS].astype(float))


def predict_proba(X_scaled: np.ndarray, model) -> float:
    return float(model.predict_proba(X_scaled)[0][1])


def classify_risk(prob: float) -> str:
    if prob < THRESHOLD_LOW:  return "low"
    if prob < THRESHOLD_HIGH: return "medium"
    return "high"


# ══════════════════════════════════════════════════════════════════════════════
#  GAUGE SVG
# ══════════════════════════════════════════════════════════════════════════════

def _gauge_svg(prob: float) -> str:
    cx, cy, r = 105, 92, 74
    risk  = classify_risk(prob)
    color = {"low": "#27AE60", "medium": "#E67E22", "high": "#E74C3C"}[risk]
 
    def _pt(p):
        a = (1.0 - p) * math.pi
        return cx + r * math.cos(a), cy - r * math.sin(a)
 
    ex, ey = _pt(prob)
    lx, ly = _pt(THRESHOLD_LOW)
    hx, hy = _pt(THRESHOLD_HIGH)
    nl     = r - 14
    angle  = (1.0 - prob) * math.pi
    nx, ny = cx + nl * math.cos(angle), cy - nl * math.sin(angle)
 
    arc_d = (
        f"M {cx - r:.1f} {cy:.1f} A {r} {r} 0 0 1 {ex:.1f} {ey:.1f}"
    ) if prob > 0.005 else ""
 
    def _tick(p, clr):
        tx, ty = _pt(p)
        a = (1.0 - p) * math.pi
        ox, oy = tx + 10 * math.cos(a), ty - 10 * math.sin(a)
        return (
            f'<line x1="{tx:.1f}" y1="{ty:.1f}" x2="{ox:.1f}" y2="{oy:.1f}" '
            f'stroke="{clr}" stroke-width="2.5" stroke-linecap="round"/>'
        )
 
    colored_arc = (
        f'<path d="{arc_d}" stroke="{color}" stroke-width="13" fill="none" stroke-linecap="round"/>'
        if arc_d else ''
    )
 
    prob_label = s("prob_label")
    parts = [
        f'<svg viewBox="0 0 210 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:280px;display:block;margin:0 auto;">',
        f'<path d="M {cx-r:.1f} {cy} A {r} {r} 0 0 1 {cx+r:.1f} {cy}" stroke="#E0EAF0" stroke-width="13" fill="none" stroke-linecap="round"/>',
        colored_arc,
        _tick(THRESHOLD_LOW,  "#27AE60"),
        _tick(THRESHOLD_HIGH, "#E74C3C"),
        f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="4.5" fill="#fff" stroke="#27AE60" stroke-width="2.5"/>',
        f'<circle cx="{hx:.1f}" cy="{hy:.1f}" r="4.5" fill="#fff" stroke="#E74C3C" stroke-width="2.5"/>',
        f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#1B3A52" stroke-width="3" stroke-linecap="round"/>',
        f'<circle cx="{cx}" cy="{cy}" r="6" fill="#1B3A52"/>',
        f'<text x="{cx}" y="{cy+22:.0f}" font-size="15" font-weight="700" fill="{color}" text-anchor="middle">{prob*100:.1f}%</text>',
        f'<text x="{lx:.0f}" y="112" font-size="8" fill="#27AE60" text-anchor="middle">23.3%</text>',
        f'<text x="{hx:.0f}" y="112" font-size="8" fill="#E74C3C" text-anchor="middle">75%</text>',
        f'<text x="{cx}" y="126" font-size="8" fill="#7A9EAF" text-anchor="middle">{prob_label}</text>',
        '</svg>'
    ]
    return ''.join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def render_header():
    st.markdown(f"""
    <div class="mc-header">
        <h1 class="mc-title">MindCheck</h1>
        <p class="mc-tagline">{s("tagline")}</p>
    </div>
    """, unsafe_allow_html=True)


def render_step_indicator(current: int):
    labels = [s("step1_label"), s("step2_label"), s("step3_label")]
    circles, lines = [], []
    for i in range(1, 4):
        if   i < current: cls, sym = "done",    "✓"
        elif i == current: cls, sym = "active",  str(i)
        else:              cls, sym = "pending", str(i)
        circles.append(f'<div class="step-circle {cls}">{sym}</div>')
        if i < 3:
            line_cls = "done" if i < current else ""
            lines.append(f'<div class="step-line {line_cls}"></div>')

    circ_html = "".join(
        c + (lines[i] if i < len(lines) else "")
        for i, c in enumerate(circles)
    )
    label_html = ""
    for i, lbl in enumerate(labels, start=1):
        cls = "done" if i < current else ("active" if i == current else "")
        label_html += f'<div class="step-label {cls}">{lbl}</div>'
        if i < 3:
            label_html += '<div class="step-label-spacer"></div>'

    st.markdown(f"""
    <div class="step-row">{circ_html}</div>
    <div class="step-label-row">{label_html}</div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Personal
# ══════════════════════════════════════════════════════════════════════════════

IKON_USER = """<svg xmlns="http://www.w3.org/2000/svg" width="34" height="34" fill="#1B3A52" viewBox="0 0 256 256"><path d="M198,112a6,6,0,0,1-6,6H152a6,6,0,0,1,0-12h40A6,6,0,0,1,198,112Zm-6,26H152a6,6,0,0,0,0,12h40a6,6,0,0,0,0-12Zm38-82V200a14,14,0,0,1-14,14H40a14,14,0,0,1-14-14V56A14,14,0,0,1,40,42H216A14,14,0,0,1,230,56Zm-12,0a2,2,0,0,0-2-2H40a2,2,0,0,0-2,2V200a2,2,0,0,0,2,2H216a2,2,0,0,0,2-2ZM133.81,166.51a6,6,0,0,1-11.62,3C119.34,158.38,108.08,150,96,150s-23.33,8.38-26.19,19.5a6,6,0,0,1-11.62-3A38,38,0,0,1,76.78,143a30,30,0,1,1,38.45,0A38,38,0,0,1,133.81,166.51ZM96,138a18,18,0,1,0-18-18A18,18,0,0,0,96,138Z"></path></svg>"""
IKON_ACAD = """
    <svg xmlns="http://www.w3.org/2000/svg" width="34" height="34" fill="#1B3A52" viewBox="0 0 256 256">
        <path d="M128,70A30,30,0,1,0,98,40,30,30,0,0,0,128,70Zm0-48a18,18,0,1,1-18,18A18,18,0,0,1,128,22Zm88.88,113.42L171.67,84.16A30,30,0,0,0,149.17,74H106.83a30,30,0,0,0-22.5,10.15L39.12,135.42A18,18,0,0,0,64.46,161l21.11-16.93L67.44,212.92a18,18,0,0,0,32.75,14.94L128,180l27.81,47.91a18,18,0,0,0,32.75-14.94l-18.13-68.87L191.54,161a18,18,0,0,0,25.34-25.56Zm-8.63,16.82a6,6,0,0,1-8.49,0,4.15,4.15,0,0,0-.49-.44l-35.51-28.48a6,6,0,0,0-9.56,6.2l22.87,86.93a7.66,7.66,0,0,0,.37,1,6,6,0,0,1-10.88,5.07,4.37,4.37,0,0,0-.25-.48L133.19,165a6,6,0,0,0-10.38,0L89.69,222.05a4.37,4.37,0,0,0-.25.48,6,6,0,0,1-10.88-5.07,7.66,7.66,0,0,0,.37-1l22.87-86.93A6,6,0,0,0,99.27,123,6.07,6.07,0,0,0,96,122a6,6,0,0,0-3.76,1.32L56.73,151.8a4.15,4.15,0,0,0-.49.44,6,6,0,0,1-8.49-8.49l.26-.27L93.33,92.09A18,18,0,0,1,106.83,86h42.34a18,18,0,0,1,13.5,6.09L208,143.48l.26.27A6,6,0,0,1,208.25,152.24Z"></path>
    </svg>
    """
IKON_LIFE = """
    <svg xmlns="http://www.w3.org/2000/svg" width="34" height="34" fill="#1B3A52" viewBox="0 0 256 256">
        <path d="M230.26,168.42l-28.62-14.31A49.72,49.72,0,0,1,174,109.39V80a6,6,0,0,0-6-6,50.06,50.06,0,0,1-50-50,6,6,0,0,0-9.62-4.78l-77,58.41-.15.11A14,14,0,0,0,30.1,98.53L143.82,212.24a6,6,0,0,0,4.24,1.76H224a14,14,0,0,0,14-14V180.94A13.94,13.94,0,0,0,230.26,168.42ZM226,200a2,2,0,0,1-2,2H150.54L38.59,90A2,2,0,0,1,38,88.52a2,2,0,0,1,.69-1.41L53.05,76.22l40,40a6,6,0,0,0,8.49-8.48L62.71,68.91,107,35.3a62.13,62.13,0,0,0,55,50.41v23.68a61.65,61.65,0,0,0,34.27,55.45l28.62,14.32a2,2,0,0,1,1.11,1.78ZM70.8,182H32a6,6,0,0,1,0-12H70.8a6,6,0,1,1,0,12Zm38,26a6,6,0,0,1-6,6H48a6,6,0,0,1,0-12h54.8A6,6,0,0,1,108.8,208Z"></path>
    </svg>
    """
IKON_SWORD = """<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" fill="#CBD5E1" viewBox="0 0 256 256"><path d="M216,36H152a4,4,0,0,0-3.17,1.56L82.05,124.38,69.19,111.52a12,12,0,0,0-17,0L39.52,124.21a12,12,0,0,0,0,17L62.34,164,31.52,194.83a12,12,0,0,0,0,17L44.2,224.49a12,12,0,0,0,17,0L92,193.67l22.81,22.82a12,12,0,0,0,17,0l12.69-12.7a12,12,0,0,0,0-17L131.62,174l86.82-66.79A4,4,0,0,0,220,104V40A4,4,0,0,0,216,36Z"></path></svg>"""

IKON_PROJECT = """
    <svg xmlns="http://www.w3.org/2000/svg" width="70" height="70" fill="#CBD5E1" viewBox="0 0 256 256">
        <path d="M98,136a6,6,0,0,1,6-6h64a6,6,0,0,1,0,12H104A6,6,0,0,1,98,136Zm6-26h64a6,6,0,0,0,0-12H104a6,6,0,0,0,0,12Zm126,82a30,30,0,0,1-30,30H88a30,30,0,0,1-30-30V64a18,18,0,0,0-36,0c0,6.76,5.58,11.19,5.64,11.23A6,6,0,1,1,20.4,84.8C20,84.48,10,76.85,10,64A30,30,0,0,1,40,34H176a30,30,0,0,1,30,30V170h10a6,6,0,0,1,3.6,1.2C220,171.52,230,179.15,230,192Zm-124,0c0-6.76-5.59-11.19-5.64-11.23A6,6,0,0,1,104,170h90V64a18,18,0,0,0-18-18H64a29.82,29.82,0,0,1,6,18V192a18,18,0,0,0,36,0Zm112,0a14.94,14.94,0,0,0-4.34-10H115.88A24.83,24.83,0,0,1,118,192a29.87,29.87,0,0,1-6,18h88A18,18,0,0,0,218,192Z"></path>
    </svg>
    """
IKON_DEVELOPER = """
    <svg xmlns="http://www.w3.org/2000/svg" width="70" height="70" fill="#CBD5E1" viewBox="0 0 256 256">
        <path d="M128,26A102,102,0,1,0,230,128,102.12,102.12,0,0,0,128,26ZM71.44,198a66,66,0,0,1,113.12,0,89.8,89.8,0,0,1-113.12,0ZM94,120a34,34,0,1,1,34,34A34,34,0,0,1,94,120Zm99.51,69.64a77.53,77.53,0,0,0-40-31.38,46,46,0,1,0-51,0,77.53,77.53,0,0,0-40,31.38,90,90,0,1,1,131,0Z"></path>
    </svg>
    """


def render_step1():
    prev = st.session_state.form_data

    st.markdown(f"""
    <div class="mc-card">
        <div class="mc-card-header">{IKON_USER}<div class="mc-card-title">{s("step1_title")}</div></div>
        <div class="mc-card-sub">{s("step1_sub")}</div>
    </div>
    """, unsafe_allow_html=True)

    gender_opts = [s("male"), s("female")]
    gender_sel  = st.radio(s("gender"), gender_opts,
                           index=prev.get("_gender_idx", 0), horizontal=True)
    gender_val  = 0 if gender_sel == s("male") else 1
    st.markdown("<hr class='mc-divider'>", unsafe_allow_html=True)

    edu_map = {s("bachelor"): 0, s("master"): 1, s("doctorate"): 2}
    edu_age_min = {s("bachelor"): 17, s("master"): 21, s("doctorate"): 24}
    edu_idx = list(edu_map.values()).index(prev.get("Education Level", 0))
    edu_sel = st.selectbox(s("edu_level"), list(edu_map.keys()), index=edu_idx)
    st.markdown("<hr class='mc-divider'>", unsafe_allow_html=True)

    age_min     = edu_age_min[edu_sel]
    age_default = max(int(prev.get("Age", age_min)), age_min)
    age_val = st.number_input(s("age"), min_value=age_min, max_value=100,
                              value=age_default, step=1, help=s("age_help"))
    st.markdown("<hr class='mc-divider'>", unsafe_allow_html=True)

    city_default = prev.get("City", CITIES[0])
    if city_default not in CITIES: city_default = CITIES[0]
    city_sel = st.selectbox(s("city"), CITIES,
                            index=CITIES.index(city_default), help=s("city_help"))
    st.markdown("<hr class='mc-divider'>", unsafe_allow_html=True)

    if st.button(s("btn_next"), type="primary", use_container_width=True):
        st.session_state.form_data.update({
            "Gender":          gender_val,
            "Age":             min(int(age_val), 35),
            "City":            city_sel,
            "Education Level": edu_map[edu_sel],
            "_gender_idx":     gender_opts.index(gender_sel),
        })
        st.session_state.step = 2
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Academic
# ══════════════════════════════════════════════════════════════════════════════

def render_step2():
    prev = st.session_state.form_data

    st.markdown(f"""
    <div class="mc-card">
        <div class="mc-card-header">{IKON_ACAD}<div class="mc-card-title">{s("step2_title")}</div></div>
        <div class="mc-card-sub">{s("step2_sub")}</div>
    </div>
    """, unsafe_allow_html=True)

    ap_val = st.slider(s("acad_press"), 1, 5, int(prev.get("Academic Pressure", 3)), help=s("acad_help"))
    st.caption(f"{'🟦'*ap_val}{'⬜'*(5-ap_val)}  **{ap_val} / 5**")
    st.markdown("<hr class='mc-divider'>", unsafe_allow_html=True)

    cgpa_val = st.slider(s("cgpa"), 1.0, 10.0, float(prev.get("CGPA", 5.0)),
                         step=0.01, help=s("cgpa_help"))
    st.markdown("<hr class='mc-divider'>", unsafe_allow_html=True)

    ss_val = st.slider(s("study_sat"), 1, 5, int(prev.get("Study Satisfaction", 3)), help=s("study_help"))
    st.caption(f"{'🟩'*ss_val}{'⬜'*(5-ss_val)}  **{ss_val} / 5**")
    st.markdown("<hr class='mc-divider'>", unsafe_allow_html=True)

    wsh_val = st.slider(s("study_hours"), 0, 16, int(prev.get("Work/Study Hours", 6)))
    st.caption(s("study_hours_caption").format(v=wsh_val))
    st.markdown("<hr class='mc-divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button(s("btn_back"), use_container_width=True):
            st.session_state.step = 1; st.rerun()
    with col2:
        if st.button(s("btn_next"), type="primary", use_container_width=True):
            st.session_state.form_data.update({
                "Academic Pressure":  ap_val,
                "CGPA":               max(float(cgpa_val), 5.0),
                "Study Satisfaction": ss_val,
                "Work/Study Hours":   wsh_val,
            })
            st.session_state.step = 3; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Lifestyle
# ══════════════════════════════════════════════════════════════════════════════

def render_step3():
    prev = st.session_state.form_data

    st.markdown(f"""
    <div class="mc-card">
        <div class="mc-card-header">{IKON_LIFE}<div class="mc-card-title">{s("step3_title")}</div></div>
        <div class="mc-card-sub">{s("step3_sub")}</div>
    </div>
    """, unsafe_allow_html=True)

    sleep_map = {
        s("sleep_lt5"): 0, s("sleep_56"): 1,
        s("sleep_78"):  2, s("sleep_gt8"): 3,
    }
    sleep_idx = list(sleep_map.values()).index(prev.get("Sleep Duration", 2))
    sleep_sel = st.selectbox(s("sleep"), list(sleep_map.keys()), index=sleep_idx)
    st.markdown("<hr class='mc-divider'>", unsafe_allow_html=True)

    diet_map = {s("diet_unhealthy"): 0, s("diet_moderate"): 1, s("diet_healthy"): 2}
    diet_idx = list(diet_map.values()).index(prev.get("Dietary Habits", 1))
    diet_sel = st.selectbox(s("diet"), list(diet_map.keys()), index=diet_idx)
    st.markdown("<hr class='mc-divider'>", unsafe_allow_html=True)

    fs_val = st.slider(s("fin_stress"), 1, 5, int(prev.get("Financial Stress", 2)), help=s("fin_help"))
    st.caption(f"{'🟧'*fs_val}{'⬜'*(5-fs_val)}  **{fs_val} / 5**")
    st.markdown("<hr class='mc-divider'>", unsafe_allow_html=True)

    fam_opts = [s("no"), s("yes")]
    fam_idx  = int(prev.get("Family History of Mental Illness", 0))
    fam_sel  = st.radio(s("fam_history"), fam_opts, index=fam_idx,
                        horizontal=True, help=s("fam_help"))
    fam_val  = 1 if fam_sel == s("yes") else 0
    st.markdown("<hr class='mc-divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button(s("btn_back"), use_container_width=True):
            st.session_state.step = 2; st.rerun()
    with col2:
        if st.button(s("btn_result"), type="primary", use_container_width=True):
            st.session_state.form_data.update({
                "Sleep Duration":                   sleep_map[sleep_sel],
                "Dietary Habits":                   diet_map[diet_sel],
                "Financial Stress":                 fs_val,
                "Family History of Mental Illness": fam_val,
            })
            st.session_state.pop("_risk_result", None)
            st.session_state.pop("_prob_result", None)
            st.session_state.step = 4; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Result
# ══════════════════════════════════════════════════════════════════════════════

def render_result(model, scaler, te):
    lang = st.session_state.get("lang", LANG_ID)

    if "_risk_result" not in st.session_state:
        try:
            X_scaled = preprocess(st.session_state.form_data, scaler, te)
            prob     = predict_proba(X_scaled, model)
            risk     = classify_risk(prob)
            st.session_state["_risk_result"] = risk
            st.session_state["_prob_result"] = prob
        except Exception as exc:
            import traceback
            st.error(f"{s('err_processing')}\n\n```\n{exc}\n```")
            with st.expander("🔍 Technical details"):
                st.code(traceback.format_exc(), language="python")
            if st.button(s("btn_retry"), type="primary"):
                for k in ["step", "form_data", "_risk_result", "_prob_result"]:
                    st.session_state.pop(k, None)
                st.rerun()
            return

    risk = st.session_state["_risk_result"]
    prob = st.session_state["_prob_result"]
    rec  = RECOMMENDATIONS[risk][lang]

    st.markdown(f"""
    <div class="result-wrap {risk}">
        <div class="result-title {risk}">{rec['title']}</div>
        <div class="result-sub">{s("prob_label")}</div>
    </div>
    """, unsafe_allow_html=True)

    import streamlit.components.v1 as components
    components.html(
        '<div style="display:flex;flex-direction:column;align-items:center;margin:.8rem 0 .2rem;">'
        + _gauge_svg(prob) + '</div>',
        height=150,
    )

    st.markdown(f"""
    <div class="mc-card" style="margin-top:.8rem;">
        <p style="color:#1B3A52;font-size:.93rem;line-height:1.75;margin:0;">{rec['desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f"<div class='rec-section'><div class='rec-header'>{s('rec_header')}</div>",
        unsafe_allow_html=True,
    )
    for title, points in rec["recs"]:
        items = "".join(f"<li>{p}</li>" for p in points)
        st.markdown(f"""
        <div class="rec-card"><h5>{title}</h5><ul>{items}</ul></div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if RECOMMENDATIONS[risk]["hotline"]:
        render_hotline(risk)

    st.markdown(f'<div class="disclaimer">{s("disclaimer")}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _, col2 = st.columns([2, 1])
    with col2:
        if st.button(s("btn_restart"), use_container_width=True):
            for k in ["step", "form_data", "_risk_result", "_prob_result"]:
                st.session_state.pop(k, None)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

IKON_SIDEBAR = """<svg xmlns="http://www.w3.org/2000/svg" width="52" height="52" fill="#CBD5E1" viewBox="0 0 256 256"><path d="M72,140H32a4,4,0,0,1,0-8H69.86l14.81-22.22a4,4,0,0,1,6.66,0l28.67,43,12.67-19A4,4,0,0,1,136,132h24a4,4,0,0,1,0,8H138.14l-14.81,22.22a4,4,0,0,1-6.66,0L88,119.21l-12.67,19A4,4,0,0,1,72,140ZM178,44c-21.44,0-39.92,10.19-50,27.07C117.92,54.19,99.44,44,78,44a58.07,58.07,0,0,0-58,58q0,1.06,0,2.13a4,4,0,1,0,8-.26c0-.62,0-1.24,0-1.87A50.06,50.06,0,0,1,78,52c21.11,0,38.85,11.31,46.3,29.51a4,4,0,0,0,7.4,0C139.15,63.31,156.89,52,178,52a50.06,50.06,0,0,1,50,50c0,58-86,109.46-100,117.42-8.47-4.82-43.5-25.61-69.63-54.12a4,4,0,0,0-5.9,5.4c30.72,33.52,71.9,55.89,73.63,56.82a4,4,0,0,0,3.8,0,333.81,333.81,0,0,0,52.7-36.73C218,160.47,236,130.59,236,102A58.07,58.07,0,0,0,178,44Z"></path></svg>"""
IKON_WARN   = """<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" fill="#F59E0B" viewBox="0 0 256 256"><path d="M236.8,188.09,149.35,36.22h0a24.76,24.76,0,0,0-42.7,0L19.2,188.09a23.51,23.51,0,0,0,0,23.72A24.35,24.35,0,0,0,40.55,224h174.9a24.35,24.35,0,0,0,21.33-12.19A23.51,23.51,0,0,0,236.8,188.09ZM120,104a8,8,0,0,1,16,0v40a8,8,0,0,1-16,0Zm8,88a12,12,0,1,1,12-12A12,12,0,0,1,128,192Z"></path></svg>"""


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(f'<div class="sb-logo">{IKON_SIDEBAR}</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-brand">MindCheck</div>', unsafe_allow_html=True)

        # ── Language selector ──────────────────────────────────────────────
        st.markdown('<div class="sb-section">Language</div>', unsafe_allow_html=True)
        lang_idx = LANG_OPTIONS.index(st.session_state.get("lang", LANG_ID))
        selected_lang = st.radio(
            "lang_radio", LANG_OPTIONS,
            index=lang_idx,
            label_visibility="collapsed",
        )
        if selected_lang != st.session_state.get("lang"):
            st.session_state.lang = selected_lang
            st.rerun()

        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
        st.markdown('<div class="sb-section">Menu</div>', unsafe_allow_html=True)

        pages = {
            s("nav_screening"):  "screening",
            s("nav_project"):    "about_project",
            s("nav_developer"):  "about_developer",
        }

        if "active_page" not in st.session_state:
            st.session_state.active_page = "screening"

        for label, key in pages.items():
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.active_page = key
                if key != "screening":
                    for k in ["step", "form_data", "_risk_result", "_prob_result"]:
                        st.session_state.pop(k, None)
                st.rerun()

        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:.72rem;color:#3D6275;text-align:center;padding:.5rem;line-height:1.6;">'
            f'MindCheck v1.0<br>Data Science Project<br>'
            f'<span style="color:#2A5570;">{IKON_WARN} {s("not_medical")}</span></p>',
            unsafe_allow_html=True,
        )

    return st.session_state.active_page


# ══════════════════════════════════════════════════════════════════════════════
#  ABOUT PAGES  (bilingual content tetap inline karena bersifat statis)
# ══════════════════════════════════════════════════════════════════════════════

def render_about_project():
    lang  = st.session_state.get("lang", LANG_ID)
    is_id = (lang == LANG_ID)
    is_hi = (lang == LANG_HI)

    st.markdown(f"""
    <div class="about-hero">
        <div class="about-hero-icon">{IKON_PROJECT}</div>
        <div class="about-hero-title">{s("about_project_title")}</div>
        <div class="about-hero-sub">{s("about_project_sub")}</div>
    </div>
    """, unsafe_allow_html=True)

    if is_id:
        st.markdown("""
        <div class="about-card">
            <div class="about-card-title">Deskripsi Proyek</div>
            <p>MindCheck adalah sistem skrining kesehatan mental berbasis data yang dirancang untuk membantu
            mahasiswa mendeteksi risiko depresi secara dini. Sistem ini menggunakan model
            <em>Logistic Regression</em> dengan pendekatan <em>klasifikasi risiko multilevel</em>
            berdasarkan ambang batas probabilitas. Prediksi dilakukan berdasarkan
            <strong>faktor-faktor risiko</strong> seperti tekanan akademik, kualitas tidur,
            dan stres finansial — bukan berdasarkan gejala klinis.</p>
        </div>
        <div class="about-card">
            <div class="about-card-title">Tujuan</div>
            <ul>
                <li>Menyediakan alat skrining awal yang mudah diakses oleh mahasiswa</li>
                <li>Mengklasifikasikan risiko depresi ke dalam tiga level berdasarkan probabilitas model</li>
                <li>Memberikan rekomendasi yang sesuai dengan level risiko masing-masing pengguna</li>
                <li>Mendorong mahasiswa untuk mencari bantuan profesional secara lebih proaktif</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif is_hi:
        st.markdown("""
        <div class="about-card">
            <div class="about-card-title">परियोजना विवरण</div>
            <p>MindCheck एक डेटा-आधारित मानसिक स्वास्थ्य जांच प्रणाली है जो छात्रों को अवसाद के
            जोखिम का जल्दी पता लगाने में मदद करने के लिए डिज़ाइन की गई है। यह प्रणाली
            <em>Logistic Regression</em> मॉडल का उपयोग करती है जिसमें संभाव्यता सीमाओं के आधार पर
            <em>बहुस्तरीय जोखिम वर्गीकरण</em> दृष्टिकोण है। भविष्यवाणियां शैक्षणिक दबाव, नींद
            की गुणवत्ता, और वित्तीय तनाव जैसे <strong>जोखिम कारकों</strong> पर आधारित हैं।</p>
        </div>
        <div class="about-card">
            <div class="about-card-title">उद्देश्य</div>
            <ul>
                <li>छात्रों के लिए आसानी से सुलभ प्रारंभिक जांच उपकरण प्रदान करना</li>
                <li>मॉडल संभावना के आधार पर अवसाद जोखिम को तीन स्तरों में वर्गीकृत करना</li>
                <li>प्रत्येक उपयोगकर्ता के जोखिम स्तर के अनुसार सुझाव प्रदान करना</li>
                <li>छात्रों को अधिक सक्रिय रूप से पेशेवर सहायता लेने के लिए प्रोत्साहित करना</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="about-card">
            <div class="about-card-title">Project Description</div>
            <p>MindCheck is a data-driven mental health screening system designed to help students
            detect depression risk early. It uses a <em>Logistic Regression</em> model with a
            <em>multilevel risk classification</em> approach based on probability thresholds.
            Predictions are based on <strong>risk factors</strong> such as academic pressure,
            sleep quality, and financial stress — not on clinical symptoms.</p>
        </div>
        <div class="about-card">
            <div class="about-card-title">Objectives</div>
            <ul>
                <li>Provide an easily accessible early screening tool for students</li>
                <li>Classify depression risk into three levels based on model probability</li>
                <li>Provide recommendations tailored to each user's risk level</li>
                <li>Encourage students to seek professional help more proactively</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Risk classification — same structure, language-aware labels
    if is_id:
        low_lbl     = "🟢 Risiko Rendah"
        med_lbl     = "🟡 Risiko Sedang"
        high_lbl    = "🔴 Risiko Tinggi"
        zone_lbl    = "Zona transisi"
        model_title = "Model & Klasifikasi Risiko"
        model_desc  = ("Model yang digunakan adalah <strong>Logistic Regression</strong> yang dilatih pada dataset mahasiswa "
                       "dengan 12 fitur prediktor. Klasifikasi risiko ditentukan menggunakan <strong>dua ambang batas probabilitas</strong>.")
    elif is_hi:
        low_lbl     = "🟢 कम जोखिम"
        med_lbl     = "🟡 मध्यम जोखिम"
        high_lbl    = "🔴 उच्च जोखिम"
        zone_lbl    = "संक्रमण क्षेत्र"
        model_title = "मॉडल और जोखिम वर्गीकरण"
        model_desc  = ("उपयोग किया गया मॉडल <strong>Logistic Regression</strong> है जिसे 12 भविष्यसूचक विशेषताओं "
                       "वाले छात्र डेटासेट पर प्रशिक्षित किया गया है। जोखिम वर्गीकरण <strong>दो संभाव्यता सीमाओं</strong> का उपयोग करके निर्धारित किया जाता है।")
    else:
        low_lbl     = "🟢 Low Risk"
        med_lbl     = "🟡 Medium Risk"
        high_lbl    = "🔴 High Risk"
        zone_lbl    = "Transition zone"
        model_title = "Model & Risk Classification"
        model_desc  = ("The model used is <strong>Logistic Regression</strong> trained on a student dataset with 12 predictor features. "
                       "Risk classification is determined using <strong>two probability thresholds</strong>.")

    st.markdown(f"""
    <div class="about-card">
        <div class="about-card-title">{model_title}</div>
        <p>{model_desc}</p>
        <div class="threshold-row">
            <div class="threshold-badge low">
                <div class="tb-label">{low_lbl}</div>
                <div class="tb-range">P &lt; 33.65%</div>
                <div class="tb-basis">Recall ≥ 0.95</div>
            </div>
            <div class="threshold-badge medium">
                <div class="tb-label">{med_lbl}</div>
                <div class="tb-range">33.65% ≤ P &lt; 70.03%</div>
                <div class="tb-basis">{zone_lbl}</div>
            </div>
            <div class="threshold-badge high">
                <div class="tb-label">{high_lbl}</div>
                <div class="tb-range">P ≥ 70.03%</div>
                <div class="tb-basis">Precision ≥ 0.90</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if is_id:
        tech_title   = "Teknologi yang Digunakan"
        source_title = "Sumber Dataset"
        note_label   = "Catatan Konteks"
        note_text    = ("Dataset ini dikumpulkan dari para mahasiswa di <strong>India</strong>. "
                        "Model mungkin paling akurat untuk populasi dengan karakteristik serupa.")
        name_lbl     = "Nama"
        source_lbl   = "Sumber"
    elif is_hi:
        tech_title   = "प्रयुक्त तकनीक"
        source_title = "डेटासेट स्रोत"
        note_label   = "संदर्भ नोट"
        note_text    = ("यह डेटासेट <strong>भारत</strong> के छात्रों से एकत्र किया गया था। "
                        "मॉडल समान विशेषताओं वाली आबादी के लिए सबसे सटीक हो सकता है।")
        name_lbl     = "नाम"
        source_lbl   = "स्रोत"
    else:
        tech_title   = "Tech Stack"
        source_title = "Dataset Source"
        note_label   = "Context Note"
        note_text    = ("This dataset was collected from students in <strong>India</strong>. "
                        "The model may be most accurate for populations with similar characteristics.")
        name_lbl     = "Name"
        source_lbl   = "Source"

    st.markdown(f"""
    <div class="about-card">
        <div class="about-card-title">{tech_title}</div>
        <div>
            <span class="tech-pill">🐍 Python 3.11</span>
            <span class="tech-pill">📊 Scikit-learn</span>
            <span class="tech-pill">🎛️ Streamlit</span>
            <span class="tech-pill">🐼 Pandas</span>
            <span class="tech-pill">🔢 NumPy</span>
            <span class="tech-pill">🏷️ Category Encoders</span>
            <span class="tech-pill">📦 Pickle</span>
            <span class="tech-pill">☁️ Streamlit Cloud</span>
        </div>
    </div>
    <div class="about-card">
        <div class="about-card-title">{source_title}</div>
        <p><strong>{name_lbl}:</strong> Student Depression Dataset</p>
        <p><strong>{source_lbl}:</strong>
            <a href="https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset"
               target="_blank" style="color:#1D6F8E;font-weight:500;">Kaggle — adilshamim8</a>
        </p>
        <p style="margin-top:.6rem;font-size:.82rem;color:#E67E22;">
            ⚠️ <strong>{note_label}:</strong> {note_text}
        </p>
    </div>
    <div class="disclaimer">{s("disclaimer")}</div>
    """, unsafe_allow_html=True)


def render_about_developer():
    lang  = st.session_state.get("lang", LANG_ID)
    is_id = (lang == LANG_ID)
    is_hi = (lang == LANG_HI)

    st.markdown(f"""
    <div class="about-hero">
        <div class="about-hero-icon">{IKON_DEVELOPER}</div>
        <div class="about-hero-title">{s("about_dev_title")}</div>
        <div class="about-hero-sub">{s("about_dev_sub")}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="dev-card">
        <div class="dev-avatar">{IKON_SWORD}</div>
        <div class="dev-name">Arthur Pendragon</div>
        <div class="dev-role">Data Scientist &nbsp;·&nbsp; Machine Learning Enthusiast</div>
        <div class="dev-links" style="text-align:center;">
            <a href="https://www.linkedin.com/in/arthurpendragon" target="_blank">🔗 LinkedIn</a>
            <a href="mailto:thedumbestknightever@gmail.com">✉️ thedumbestknightever@gmail.com</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if is_id:
        about_me      = ("Lahir ke dunia tanpa sempat memberikan persetujuan, namun kini memilih untuk menetap dengan penuh kesadaran. "
                         "Berawal dari rasa penasaran masa kecil terhadap mesin dan logika, kini saya melangkah ke fase dewasa "
                         "dengan membangun solusi nyata. Ini adalah proyek pertama saya — sebuah upaya untuk memberi makna di antara baris kode dan data.")
        about_title   = "Tentang Saya"
        skills_title  = "Keahlian"
        contact_title = "Hubungi Saya"
        contact_desc  = "Tertarik untuk berdiskusi, berkolaborasi, atau sekadar menyapa? Jangan ragu untuk menghubungi saya."
    elif is_hi:
        about_me      = ("बिना सहमति के इस दुनिया में आए, लेकिन अब पूरी जागरूकता के साथ यहाँ रहने का चुनाव किया है। "
                         "बचपन में मशीनों और तर्क के प्रति जो जिज्ञासा थी, वह अब वयस्कता में वास्तविक समाधान बनाने में बदल गई है। "
                         "यह मेरा पहला प्रोजेक्ट है — कोड और डेटा की पंक्तियों के बीच अर्थ खोजने का एक प्रयास।")
        about_title   = "मेरे बारे में"
        skills_title  = "कौशल"
        contact_title = "संपर्क करें"
        contact_desc  = "चर्चा, सहयोग, या बस नमस्ते कहने में रुचि है? बेझिझक संपर्क करें।"
    else:
        about_me      = ("Cast into existence without prior consent, yet now choosing to reside here with full intent. "
                         "What began as a childhood fascination with machines and logic has evolved into a purposeful step into adulthood. "
                         "This is my first project — a dedicated effort to find meaning between lines of code and data.")
        about_title   = "About Me"
        skills_title  = "Skills"
        contact_title = "Get In Touch"
        contact_desc  = "Interested in discussing, collaborating, or just saying hi? Feel free to reach out."

    st.markdown(f"""
    <div class="about-card">
        <div class="about-card-title">{about_title}</div>
        <p>{about_me}</p>
    </div>
    <div class="about-card">
        <div class="about-card-title">{skills_title}</div>
        <div>
            <span class="tech-pill">Machine Learning</span>
            <span class="tech-pill">Data Analysis</span>
            <span class="tech-pill">Python</span>
            <span class="tech-pill">Statistical Modeling</span>
            <span class="tech-pill">Data Visualization</span>
            <span class="tech-pill">Streamlit</span>
            <span class="tech-pill">Scikit-learn</span>
        </div>
    </div>
    <div class="about-card">
        <div class="about-card-title">{contact_title}</div>
        <p>{contact_desc}</p>
        <div style="margin-top:.8rem;">
            <a href="https://www.linkedin.com/in/arthurpendragon" target="_blank"
               style="display:inline-flex;align-items:center;gap:.4rem;background:#0A66C2;color:#fff;
               text-decoration:none;padding:.5rem 1.1rem;border-radius:8px;font-size:.85rem;
               font-weight:600;margin-right:.6rem;margin-bottom:.4rem;">LinkedIn</a>
            <a href="mailto:thedumbestknightever@gmail.com"
               style="display:inline-flex;align-items:center;gap:.4rem;background:#1D6F8E;color:#fff;
               text-decoration:none;padding:.5rem 1.1rem;border-radius:8px;font-size:.85rem;
               font-weight:600;margin-bottom:.4rem;">Email</a>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    inject_css()

    if "lang" not in st.session_state:
        st.session_state.lang = LANG_ID

    active_page = render_sidebar()

    if active_page == "about_project":
        render_about_project(); return
    if active_page == "about_developer":
        render_about_developer(); return

    render_header()

    if "step"      not in st.session_state: st.session_state.step      = 1
    if "form_data" not in st.session_state: st.session_state.form_data = {}

    step = st.session_state.step

    try:
        model, scaler, te = load_models()
    except FileNotFoundError as e:
        st.error(f"{s('err_file')}\n\n`{e}`")
        st.stop()

    if step <= 3:
        render_step_indicator(step)

    if   step == 1: render_step1()
    elif step == 2: render_step2()
    elif step == 3: render_step3()
    elif step == 4: render_result(model, scaler, te)


if __name__ == "__main__":
    main()