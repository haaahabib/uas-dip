# Laporan Proyek Data, Informasi, dan Pengetahuan - Muhammad Habibulloh

## Domain Proyek

Diabetes merupakan penyakit metabolik kronis yang ditandai dengan peningkatan kadar glukosa darah atau gula darah, yang lama-kelamaan dapat menyebabkan kerusakan serius pada jantung, pembuluh darah, mata, ginjal, dan saraf. Menurut laporan WHO, sekitar 830 juta orang di seluruh dunia mengidap diabetes [[1]](https://www.who.int/health-topics/diabetes). Untuk mempercepat diagnosis dan mengetahui penyakit diabetes seseorang diperlukan deteksi dini. Deteksi dini penting untuk mencegah komplikasi kerusakan serius pada organ-organ manusia dan meningkatkan kualitas hidup. Proyek ini bertujuan membangun model prediktif untuk mengidentifikasi risiko diabetes berdasarkan parameter klinis pasien.

## Referensi:

1. Dataset: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data) (768 sampel).
2. I. Tasin, T. U. Nabil, S. Islam, and R. Khan, "Diabetes prediction using machine learning and explainable AI techniques," Healthcare Technology Letters, vol. 10, no. 1, pp. 1-10, Feb. 2023, doi: 10.1049/htl2.12039. [PubMed](https://pmc.ncbi.nlm.nih.gov/articles/PMC10107388/)
3. World Health Organization, [WHO, 2023](https://www.who.int/health-topics/diabetes)

## Business Understanding

### Problem Statements

- Prevalensi diabetes di Indonesia terus meningkat, dari 10.9% menjadi 11.7% pada tahun 2023 [Kemenkes](https://rspermatajonggol.com/ini-5-sebab-diabetes-tumbuh-subur-di-indonesia/)
- Keterlambatan diagnosis diabetes dapat menyebabkan komplikasi serius seperti jantung, pembuluh darah, mata, ginjal, dan saraf.
- Sistem prediksi berbasis machine learning dapat membantu tenaga medis dalam melakukan skrining awal dan mengidentifikasi pasien yang berisiko tinggi, sehingga diagnosis dapat dilakukan lebih cepat.
- Dataset diabetes seringkali memiliki data imbalance, di mana jumlah pasien non-diabetes jauh lebih banyak daripada pasien diabetes. Hal ini dapat menyebabkan model cenderung memprediksi pasien sebagai non-diabetes, sehingga akurasi prediksi untuk pasien diabetes menjadi rendah.

### Goals

- Membangun model klasifikasi machine learning dengan akurasi minimal 75% pada data test.
- Menyediakan sistem yang dapat memberikan probabilitas risiko diabetes dan rekomendasi untuk pemeriksaan lanjutan, sehingga dapat digunakan sebagai alat bantu second opinion bagi tenaga medis.
- Menangani data imbalance menggunakan teknik oversampling SMOTE untuk meningkatkan performa model dalam memprediksi pasien diabetes.

### Solution Statements

Goal 1: Membangun model klasifikasi dengan akurasi minimal 75% pada data test.
- Solusi 1: Membandingkan performa algoritma Logistic Regression, Random Forest, dan Gradient Boosting. Alasannya Logistic Regression adalah algoritma dasar yang mudah diinterpretasi, Random Forest memiliki performa yang baik dan robust, sedangkan Gradient Boosting dapat menghasilkan model yang akurat. Perbandingan ini bertujuan untuk memilih model yang paling optimal untuk dataset diabetes.
- Solusi 2: Meningkatkan performa model terbaik dengan Hyperparameter Tuning. Alasannya, hyperparameter tuning memungkinkan optimasi parameter model untuk mencapai performa terbaik. 

Goal 2: Menyediakan sistem yang dapat memberikan probabilitas risiko diabetes dan rekomendasi untuk pemeriksaan lanjutan, sehingga dapat digunakan sebagai alat bantu second opinion bagi tenaga medis.

Goal 3: Menangani data imbalance untuk meningkatkan performa model dalam memprediksi pasien diabetes. Dengan menerapkan teknik SMOTE (Synthetic Minority Over-sampling Technique) pada data latih. Alasannya adalah SMOTE akan menyeimbangkan jumlah data pasien diabetes dan non-diabetes, sehingga model dapat mempelajari pola dari kedua kelas dengan lebih baik dan meningkatkan performa, khususnya dalam mendeteksi pasien diabetes.
    
## Data Understanding

Pima Indians Diabetes Database [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/code) (768 sampel)

### Variabel-variabel pada Pima Indians Diabetes Database adalah sebagai berikut:

- Pregnancies: Jumlah kehamilan
- Glucose: Konsentrasi glukosa plasma 2 jam dalam tes toleransi glukosa oral
- BloodPressure: Tekanan darah diastolik (mm Hg)
- SkinThickness: Ketebalan lipatan kulit trisep (mm)
- Insulin: Insulin serum 2 jam (mu U/ml)
- BMI: Indeks massa tubuh (berat dalam kg/(tinggi dalam m)^2)
- DiabetesPedigreeFunction: Variabel yang menunjukkan riwayat diabetes dalam keluarga
- Age: Usia dalam tahun
- Outcome: Variabel kelas (0 atau 1), di mana 0 menunjukkan tidak menderita diabetes dan 1 menunjukkan menderita diabetes.

### Distribusi Pasien Diabetes dan Non-Diabetes pada Pima Indians Diabetes Database

![image](https://github.com/user-attachments/assets/f4903af7-5f29-4666-9d7c-7486e56f7379)

## Data Preparation

1. Missing Value Handling:
Nilai `0` pada kolom `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, dan `BMI` diganti dengan `NaN` karena dianggap tidak valid secara medis. Ini dilakukan untuk mengidentifikasi *missing value* yang disamarkan.

Penanganan `NaN` kemudian dilakukan sebagai berikut:
* Untuk kolom `Glucose`, `BloodPressure`, `SkinThickness`, dan `BMI`, nilai `NaN` diisi dengan **nilai rata-rata (mean)** dari setiap kolom, sesuai dengan pendekatan umum pada paper referensi.
* Khusus untuk kolom `Insulin`, nilai `NaN` diisi menggunakan **XGBoost Regressor**. Model ini dilatih pada data `Insulin` yang tersedia untuk memprediksi nilai yang hilang, mereplikasi pendekatan semi-supervised dari paper.

2. Data Splitting:
Data dibagi menjadi data train (`80%`) dan data test (`20%`) menggunakan fungsi `train_test_split`.  
Data train digunakan untuk melatih model, sedangkan data test digunakan untuk mengevaluasi performa model pada data yang belum pernah dilihat sebelumnya. Pembagian data ini penting untuk memastikan kemampuan generalisasi model.

3. Scaling:
**Min-Max Normalization** digunakan untuk menyamakan skala antar fitur, menskalakan nilai-nilai ke dalam rentang [0, 1]. Normalisasi ini penting karena algoritma Machine Learning peka terhadap skala fitur dan metode ini konsisten dengan yang digunakan dalam paper referensi.

4. Penanganan Data Tidak Seimbang (SMOTE):
Menerapkan teknik `SMOTE` (Synthetic Minority Over-sampling Technique) untuk mengatasi ketidakseimbangan kelas pada data training.  
Ketidakseimbangan kelas dapat menyebabkan model cenderung memprediksi kelas mayoritas. `SMOTE` menghasilkan data sintetis untuk kelas minoritas sehingga distribusi kelas menjadi lebih seimbang dan meningkatkan performa model dalam memprediksi kelas minoritas.

### Feature Selection

Mengikuti metodologi paper referensi yang menggunakan Mutual Information, fitur **`DiabetesPedigreeFunction`** telah **dihapus** dari dataset. Paper mengidentifikasi fitur ini sebagai kurang signifikan, sehingga penghapusannya bertujuan untuk memfokuskan model pada fitur yang lebih relevan dan berpotensi meningkatkan kinerja.

## Modeling

Pada proses pemodelan, digunakan tiga algoritma machine learning:

#### 1. Logistic Regression
- **Tahapan:** Inisialisasi model dengan `LogisticRegression(max_iter=1000)`. Parameter `max_iter` ditingkatkan untuk memastikan konvergensi model.  
- **Kelebihan:** Sederhana, mudah diinterpretasi, dan efisien untuk dataset yang besar.  
- **Kekurangan:** Asumsi linearitas, sensitif terhadap outlier, dan mungkin kurang akurat untuk data yang kompleks.  

#### 2. Random Forest
- **Tahapan:** Inisialisasi model dengan `RandomForestClassifier()`.  
- **Kelebihan:** Akurasi tinggi, robust terhadap outlier dan missing value, dapat menangani data non-linear.  
- **Kekurangan:** Kompleks, sulit diinterpretasi, dan membutuhkan waktu pelatihan yang lebih lama.  

#### 3. Gradient Boosting
- **Tahapan:** Inisialisasi model dengan `GradientBoostingClassifier()`.  
- **Kelebihan:** Akurasi tinggi, dapat menangani data non-linear, dan performa yang baik pada berbagai jenis dataset.  
- **Kekurangan:** Kompleks, cenderung overfitting jika tidak di-*tuning* dengan baik, dan membutuhkan waktu pelatihan yang lebih lama.  

---

### Hyperparameter Tuning (Random Forest)

Karena `RandomForestClassifier` memberikan performa terbaik pada tahap awal, dilakukan *hyperparameter tuning* menggunakan `GridSearchCV` untuk meningkatkan performanya.

- **Parameter yang di-tuning:**
  - `n_estimators`: Jumlah pohon dalam forest (`100`, `200`)
  - `max_depth`: Kedalaman maksimum setiap pohon (`5`, `10`, `None`)
  - `min_samples_split`: Jumlah minimum sampel yang diperlukan untuk membagi node (`2`, `5`)

- **Proses:**  
  `GridSearchCV` mengevaluasi semua kombinasi parameter menggunakan *5-fold cross-validation* dan memilih kombinasi terbaik berdasarkan akurasi.

---

### Pemilihan Model Terbaik

Meskipun tuning tidak meningkatkan akurasi `RandomForestClassifier`, model ini tetap dipilih sebagai model terbaik karena:

- Akurasi awal yang tinggi dibandingkan algoritma lain.
- *Hyperparameter* default sudah cukup optimal untuk dataset ini.

## Evaluation

### Model Terbaik

Model terbaik yang dipilih adalah `Random Forest` dengan akurasi sebesar `0.77`.

### Akurasi

- Logistic Regression: 0.71  
- Random Forest: 0.74  
- Gradient Boosting: 0.70  
- Tuned Random Forest: 0.72

### Recall

| Model              | Recall (0) | Recall (1) |
|--------------------|------------|------------|
| Logistic Regression| 0.72       | 0.71       |
| Random Forest      | 0.73       | 0.76       |
| Gradient Boosting  | 0.68       | 0.75       |

Dalam tugas klasifikasi, **recall** sangat penting, terutama ketika kita ingin memastikan bahwa semua instance positif (misalnya, pasien diabetes) dapat terdeteksi dengan baik. Recall mengukur kemampuan model untuk menemukan semua kasus positif yang sebenarnya, sehingga semakin tinggi nilai recall, semakin sedikit kasus positif yang terlewat (false negatives). Dalam konteks diagnosis medis, misclassifying a positive case (e.g., misdiagnosing a diabetic patient as healthy) bisa berisiko tinggi, sehingga model dengan recall yang lebih tinggi sangat diutamakan untuk mengurangi potensi kesalahan diagnosis.

### Confusion Matrix

Keterangan:
- **True Negative (TN)**: 81 Pasien non-diabetes diprediksi benar sebagai non-diabetes
- **False Positive (FP)**: 16 Pasien non-diabetes salah diprediksi sebagai diabetes
- **False Negative (FN)**: 15 Pasien diabetes salah diprediksi sebagai non-diabetes
- **True Positive (TP)**: 42 Pasien diabetes diprediksi benar sebagai diabetes

---

### Kesimpulan
Berdasarkan metrik evaluasi yang digunakan, model `Random Forest` menunjukkan performa yang baik dalam memprediksi risiko diabetes.  


**--- Terima Kasih ---**
