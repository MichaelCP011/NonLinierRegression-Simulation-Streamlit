# Simulasi Regresi Non-Linier Interaktif
Copyright © 2025 [Michael Christia Putra]

---

Sebuah aplikasi web interaktif yang dibangun menggunakan Streamlit untuk melakukan analisis dan perbandingan beberapa model regresi non-linier. Aplikasi ini membantu pengguna menemukan model matematis yang paling cocok untuk set data mereka berdasarkan nilai *Sum of Squared Errors* (SSE) terkecil.

## Input Data
![Cuplikan layar 2025-07-05 003723](https://github.com/user-attachments/assets/4159e112-7466-4224-ad00-f2fd05241f70)

## Plotting Data
![Cuplikan layar 2025-07-05 003854](https://github.com/user-attachments/assets/366cdd08-ed5c-494e-84b5-220d49ede494)

## Model Terbaik
![Cuplikan layar 2025-07-05 003952](https://github.com/user-attachments/assets/954eac0d-c7e7-4250-b151-27680d9767a2)


## 📖 Deskripsi Proyek
Proyek ini bertujuan untuk menyediakan alat yang mudah digunakan bagi mahasiswa, peneliti, atau analis data untuk memvisualisasikan dan menganalisis hubungan non-linier dalam data. Daripada melakukan perhitungan manual yang rumit, pengguna dapat secara dinamis memasukkan data dan langsung melihat perbandingan tiga model regresi umum.

## ✨ Fitur Utama
* **Perbandingan Tiga Model**: Menganalisis dan membandingkan model berikut secara bersamaan:
    1.  **Model Eksponensial**: $y = C \cdot e^{bx}$
    2.  **Model Pangkat Sederhana**: $y = C \cdot x^b$
    3.  **Model Laju Pertumbuhan Jenuh**: $y = \frac{C \cdot x}{d + x}$
* **Input Data Dinamis**: Pengguna dapat memasukkan data melalui dua cara:
    * Input manual (copy-paste nilai x dan y).
    * Mengunggah file CSV.
* **Visualisasi Interaktif**: Menampilkan plot sebar (scatter plot) dari data asli dan plot perbandingan yang menyajikan garis dari setiap model regresi.
* **Hasil Analisis Lengkap**: Menampilkan persamaan yang dihasilkan dari setiap model, tabel perhitungan detail, dan nilai SSE.
* **Kesimpulan Otomatis**: Secara otomatis mengidentifikasi dan menyorot model terbaik (dengan SSE terkecil).

## 🛠️ Teknologi yang Digunakan
* **Python**: Bahasa pemrograman utama.
* **Streamlit**: Kerangka kerja untuk membangun aplikasi web interaktif.
* **Pandas**: Untuk manipulasi dan menampilkan data dalam bentuk tabel.
* **NumPy**: Untuk perhitungan numerik yang efisien.
* **Matplotlib**: Untuk membuat visualisasi dan plot data.

## 🚀 Instalasi dan Menjalankan Proyek
Ikuti langkah-langkah berikut untuk menjalankan aplikasi ini di komputer lokal Anda.

### 1. Prasyarat
Pastikan Anda sudah menginstal **Python 3.8** atau versi yang lebih baru.

### 2. Clone Repositori
Buka terminal atau command prompt Anda dan clone repositori ini:
```bash
git clone https://github.com/MichaelCP011/NonLinierRegression-Simulation-Streamlit.git
cd NonLinierRegression-Simulation-Streamlit
