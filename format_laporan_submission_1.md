# Laporan Proyek Machine Learning - Nama Anda

## Domain Proyek

Diabetes mellitus merupakan salah satu penyakit kronis dengan prevalensi yang terus meningkat di seluruh dunia. Berdasarkan laporan terbaru dari International Diabetes Federation (IDF), diperkirakan sebanyak 589 juta orang dewasa (usia 20–79 tahun) hidup dengan diabetes pada tahun 2024, dan angka ini diproyeksikan meningkat menjadi 853 juta pada tahun 2050 [1]. Sekitar 43% penderita diabetes dewasa tidak terdiagnosis, yang menyebabkan keterlambatan penanganan dan meningkatkan risiko komplikasi serius seperti penyakit jantung, gagal ginjal, dan kebutaan [1].
Salah satu tantangan besar dalam upaya pengendalian diabetes adalah deteksi dini, khususnya untuk tipe 2 yang berkembang secara perlahan dan seringkali tanpa gejala yang jelas. Pendekatan konvensional dalam diagnosis sering membutuhkan pemeriksaan laboratorium dan kunjungan medis berkala, yang mungkin tidak selalu dapat diakses oleh masyarakat luas, terutama di negara berkembang.
Di sinilah pendekatan machine learning (ML) berperan sebagai solusi modern. Machine learning memungkinkan sistem untuk mempelajari pola dari data medis historis — seperti kadar glukosa darah, tekanan darah, indeks massa tubuh (BMI), usia, dan jumlah kehamilan — dan kemudian digunakan untuk memprediksi risiko seseorang terkena diabetes secara otomatis dan efisien. Algoritma klasifikasi dalam ML terbukti memiliki akurasi yang tinggi dalam mengidentifikasi pasien berisiko tanpa intervensi manual secara langsung [2].
Salah satu dataset yang umum digunakan adalah Pima Indians Diabetes Dataset (PIDD), yang berisi 768 data kuantitatif dari wanita keturunan Pima Indian di AS [3]. Dataset ini telah menjadi benchmark dalam riset kesehatan prediktif, digunakan untuk melatih dan menguji berbagai model machine learning seperti Logistic Regression, Random Forest, dan Neural Network [4].
Mengadopsi pendekatan ML tidak hanya memungkinkan efisiensi dalam proses skrining, tetapi juga dapat diterapkan sebagai sistem peringatan dini (early warning system) dalam platform digital seperti aplikasi kesehatan masyarakat. Pendekatan ini sangat relevan dan strategis di era transformasi digital layanan kesehatan.

Referensi :

[1] International Diabetes Federation, “IDF Diabetes Atlas, 2024 Update,” [Online]. Available: https://diabetesatlas.org/data-by-location/global/. [Accessed: 19-May-2025].

[2] E. Alpaydin, Introduction to Machine Learning, 4th ed., Cambridge, MA: MIT Press, 2020.

[3] UCI Machine Learning Repository, “Pima Indians Diabetes Database,” [Online]. Available: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database. [Accessed: 19-May-2025].

[4] H. Aslam et al., “Comparative Analysis of Machine Learning Techniques for the Prediction of Diabetes,” Scientific Reports, vol. 13, no. 1, pp. 1–10, Jan. 2023. [Online]. Available: https://www.nature.com/articles/s41598-023-27591-9.

## Business Understanding

### Problem Statements

Bedasarkan latar belakang di atas, maka dapat disimpulkan bahwa masalah dalam pengembangan ini adalah:

1. Tingkat keterlambatan diagnosis diabetes masih tinggi, terutama pada penderita diabetes tipe 2 yang sering tidak menunjukkan gejala awal secara jelas. Akibatnya, banyak penderita baru menyadari kondisinya setelah terjadi komplikasi serius yang memerlukan penanganan lebih kompleks dan mahal.

2. Proses diagnosis konvensional membutuhkan fasilitas medis dan tenaga kesehatan, yang belum tentu mudah diakses oleh semua lapisan masyarakat, terutama di daerah terpencil atau negara dengan keterbatasan sumber daya.

3. Belum tersedia sistem deteksi dini yang bersifat otomatis, efisien, dan berbasis data kuantitatif yang dapat membantu dalam proses skrining awal risiko diabetes secara cepat dan akurat.

4. Meskipun telah tersedia data medis historis seperti kadar glukosa darah, tekanan darah, BMI, dan usia pasien, data tersebut belum dimanfaatkan secara optimal melalui teknologi machine learning untuk mengidentifikasi pola risiko diabetes secara prediktif.

5. Diperlukan pengembangan model klasifikasi berbasis machine learning yang dapat memprediksi risiko diabetes secara akurat dan dapat diintegrasikan ke dalam sistem digital kesehatan sebagai alat bantu skrining awal.

### Goals

Tujuan dari proyek ini adalah :

1. Mengembangkan model prediksi risiko diabetes menggunakan pendekatan machine learning, khususnya metode klasifikasi, berdasarkan data kuantitatif seperti kadar glukosa darah, tekanan darah, BMI, usia, dan parameter medis lainnya.

2. Meningkatkan deteksi dini terhadap potensi diabetes tipe 2, dengan memanfaatkan data historis dan algoritma yang dapat mengenali pola-pola risiko secara otomatis tanpa perlu intervensi medis langsung.

3. Menyediakan solusi skrining yang efisien, akurat, dan dapat digunakan secara luas, terutama bagi masyarakat yang memiliki akses terbatas terhadap fasilitas medis.

4. Memanfaatkan dan mengoptimalkan dataset medis (seperti Pima Indians Diabetes Dataset) sebagai dasar pengembangan dan evaluasi model prediktif diabetes berbasis machine learning.

5. Menguji dan mengevaluasi performa berbagai algoritma klasifikasi (misalnya: Logistic Regression, Decision Tree, Random Forest, Neural Network) untuk menentukan model terbaik yang dapat diintegrasikan dalam sistem pendukung keputusan di bidang kesehatan.

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah Pima Indians Diabetes Database, yang berasal dari UCI Machine Learning Repository dan juga tersedia di Kaggle. Dataset ini dikumpulkan oleh National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) dan difokuskan pada populasi wanita keturunan Pima Indian di Arizona, Amerika Serikat, yang berusia 21 tahun ke atas. Dataset ini memiliki 768 baris dengan 9 variabel yang salah satunya adalah variabel target. [Sumber : Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data)

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Pima Indians Diabetes Database ini adalah sebagai berikut:
- Pregnancies : Jumlah kehamilan
- Glucose : Konsentrasi glukosa plasma (glucose level) 2 jam dalam tes toleransi glukosa
- BloodPressure : Tekanan darah diastolik (mm Hg)
- SkinThickness : Ketebalan lipatan kulit trisep (mm)
- Insulin : Kadar insulin serum 2 jam (mu U/ml)
- BMI : Indeks massa tubuh (kg/m²)
- DiabetesPedigreeFunction : Fungsi riwayat keluarga diabetes (probabilistik)
- Age : Usia pasien (dalam tahun)
- Outcome : Label target: 0 = tidak diabetes, 1 = diabetes

### Nilai 0 atau Null pada kolom medis (Glucose, BloodPressure, SkinThickness, Insulin, dan BMI).
- Glucose : 5
- BloodPressure : 35
- SkinThickness : 227
- Insulin : 374
- BMI : 11

**Catatan : Nilai 0 pada kolom medis dianggap sebagai error karena nilai tersebut tidak mungkin terjadi pada fitur-fitur medis**

### Distribusi Fitur
![Deskripsi](https://raw.githubusercontent.com/username/repo-name/branch-name/path/to/image.png)


Berdasarkan analisis distribusi fitur, diketahui bahwa sebagian besar pasien memiliki riwayat kehamilan antara 0 hingga 3 kali, dengan jumlah yang jauh lebih sedikit pada kehamilan lebih dari 10, menunjukkan distribusi positif skew pada fitur Pregnancies. Fitur Glucose memiliki distribusi yang mendekati normal namun sedikit skew ke kanan, dengan rentang umum antara 90 hingga 150; meskipun demikian, terdapat nilai 0 yang secara medis tidak masuk akal, yang menunjukkan kemungkinan data hilang atau tidak terukur. BloodPressure memperlihatkan distribusi hampir normal, dengan mayoritas pasien memiliki tekanan darah di kisaran 60–80 mmHg, namun juga ditemukan nilai 0 yang perlu diatasi dalam tahap praproses data. Distribusi SkinThickness sangat skew ke kanan dengan banyak nilai 0, mengindikasikan bahwa pengukuran ini mungkin tidak dilakukan pada banyak pasien. Fitur Insulin memiliki distribusi yang sangat tidak merata, dengan lebih dari separuh data menunjukkan nilai 0, yang mengindikasikan kemungkinan besar data yang hilang. Pada fitur BMI, mayoritas pasien memiliki indeks massa tubuh antara 30–40, yang termasuk kategori overweight atau obesitas, dan distribusinya mendekati normal tetapi tetap mengandung beberapa nilai 0 yang tidak logis. Fitur DiabetesPedigreeFunction menunjukkan skew ke kanan, dengan sebagian besar pasien memiliki skor risiko keturunan rendah (< 0.5), sementara hanya sedikit yang menunjukkan risiko genetik tinggi. Age juga menunjukkan skew ke kanan, dengan dominasi usia pasien antara 20 hingga 40 tahun, dan hanya sedikit pasien yang berusia lebih dari 60 tahun. Sementara itu, fitur Outcome sebagai variabel target bersifat biner dengan proporsi yang tidak seimbang, yaitu sekitar 65% pasien tidak menderita diabetes (label 0) dan 35% menderita diabetes (label 1), yang menandakan adanya klasifikasi imbalance yang perlu ditangani pada tahap modeling.

### Boxplot 
![Boxplot Untuk Deteksi Outlier]("./Image/2.0.0. Deteksi Outlier.png")
## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

