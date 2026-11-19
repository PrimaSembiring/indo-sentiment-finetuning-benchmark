# Rancangan Eksperimen

## Tujuan

Membandingkan backbone Transformer pretrained dan strategi fine-tuning untuk klasifikasi
sentimen Bahasa Indonesia 3 kelas (negatif, netral, positif), lalu menentukan strategi terbaik
berdasarkan performa (F1 makro, akurasi) dan efisiensi (waktu latih, waktu inferensi,
jumlah parameter, memori GPU).

## Dataset

SmSA dari benchmark IndoNLU (Wilie dkk., 2020), diunduh langsung dari repositori resmi.

| Split | Baris asli | Setelah dibersihkan |
|---|---|---|
| Latih (`train_preprocess.tsv`) | 11.000 | 10.933 (67 duplikat dibuang) |
| Validasi (`valid_preprocess.tsv`) | 1.260 | ±1.246 (14 teks yang juga ada di data latih dibuang) |
| Uji (`test_preprocess_masked_label.tsv`) | 500 | Tidak dipakai, label disembunyikan |

Distribusi label latih timpang: positif 58%, negatif 31%, netral 10%.

Pembersihan dilakukan otomatis di notebook:
1. Buang duplikat persis (teks + label).
2. Buang teks yang memiliki label bertentangan.
3. Buang teks validasi yang juga muncul di data latih (mencegah kebocoran data).

## Protokol evaluasi

Label data uji resmi tidak tersedia, sehingga **validasi resmi yang sudah dibersihkan dipakai
sebagai held-out set**. Agar evaluasi tetap jujur:
- Tidak ada early stopping dan tidak ada pemilihan checkpoint berdasarkan validasi.
- Setiap run dilatih dengan jumlah epoch tetap, lalu model akhir dievaluasi satu kali.

## Backbone

| Kode | Checkpoint | Keterangan |
|---|---|---|
| `mbert` | `bert-base-multilingual-cased` | Baseline multibahasa |
| `distilbert` | `distilbert-base-multilingual-cased` | Versi ringkas (6 layer), untuk melihat efisiensi |
| `indobert` | `indobenchmark/indobert-base-p1` | Dilatih khusus Bahasa Indonesia (Indo4B) |
| `xlmr` | `xlm-roberta-base` | Multibahasa modern, hanya full fine-tuning |

## Strategi fine-tuning

Semua run memakai **head replacement**: classification head baru dengan 3 output.

| Strategi | Bagian yang dilatih | Learning rate |
|---|---|---|
| Full fine-tuning | Semua parameter | 2e-5 |
| Partial freezing | Embedding + separuh layer awal dibekukan; separuh layer akhir, pooler, dan head dilatih | 2e-5 |
| Linear probing | Seluruh backbone dibekukan; hanya head yang dilatih | 1e-3 |

## Matriks eksperimen (24 run)

| Backbone \ Strategi | Full | Partial | Probing |
|---|---|---|---|
| mBERT | ✓ ×3 seed | ✓ ×3 seed | — |
| DistilBERT | ✓ ×3 seed | ✓ ×3 seed | — |
| IndoBERT | ✓ ×3 seed | ✓ ×3 seed | ✓ ×3 seed |
| XLM-R | ✓ ×3 seed | — | — |

- Tiga backbone (mBERT, DistilBERT, IndoBERT) dibandingkan dengan protokol identik pada full dan partial.
- IndoBERT menjalankan ketiga strategi, sehingga pengaruh strategi bisa dilihat pada satu backbone.

## Variabel terkendali

| Variabel | Nilai |
|---|---|
| `max_length` | 128 token (p99 panjang kalimat = 79 kata) |
| Batch size latih | 32 |
| Epoch | 3 |
| Seed | 42, 123, 2024 |
| Optimizer | AdamW, weight decay 0,01 |
| Scheduler | Linear, warmup 10% dari total step |
| Presisi | fp16 (mixed precision) bila GPU tersedia |
| Data latih | Penuh (10.933 kalimat) |

## Metrik per run

- Performa: akurasi, F1 makro, F1 per kelas (negatif, netral, positif), confusion matrix.
- Efisiensi: waktu latih, waktu inferensi per kalimat (batch 1, 200 kalimat, 10 kalimat warmup),
  jumlah parameter total dan yang dilatih, memori GPU puncak.

## Hipotesis

- **H1 (backbone):** IndoBERT mencapai F1 makro tertinggi dibanding mBERT dan DistilBERT pada strategi
  yang sama, karena korpus pretraining dan kosakatanya selaras dengan Bahasa Indonesia. XLM-R diperkirakan
  bersaing dengan IndoBERT, dengan biaya parameter dan inferensi lebih besar.
- **H2 (strategi):** full fine-tuning > partial freezing > linear probing pada F1 makro. Partial freezing
  kehilangan sedikit F1 tetapi lebih hemat waktu latih dan memori; linear probing paling hemat tetapi
  F1-nya jauh lebih rendah.
- **H3 (efisiensi):** DistilBERT memiliki sekitar 40% lebih sedikit parameter dan inferensi sekitar
  1,6–2× lebih cepat dibanding mBERT, dengan penurunan F1 makro yang kecil (Sanh dkk., 2019).

## Aturan menyimpulkan

Dua konfigurasi dianggap **berbeda secara meyakinkan hanya jika selisih rata-rata F1 makro lebih
besar daripada simpangan baku antar-seed** (diambil yang terbesar dari keduanya).

## Platform

- Pengembangan dan uji cepat: Jupyter lokal (RTX 4050 Laptop 6 GB), `MODE="uji"`.
- Eksperimen resmi: Kaggle (GPU T4), `MODE="penuh"`, seluruh 24 run dalam satu sesi agar waktu
  inferensi dapat dibandingkan secara adil. Hasil lokal tidak dicampur dengan hasil Kaggle.

## Keterbatasan

- Evaluasi memakai validasi resmi karena label uji tidak tersedia.
- Hyperparameter tidak dituning per backbone; nilainya sama untuk semua agar perbandingan adil.
- Hanya satu dataset, sehingga kesimpulan belum tentu berlaku untuk tugas NLP lain.
- Waktu latih dan inferensi bergantung pada jenis GPU.## Dataset

SmSA dari benchmark IndoNLU (Wilie dkk., 2020), diunduh langsung dari repositori resmi.

| Split | Baris asli | Setelah dibersihkan |
|---|---|---|
| Latih (`train_preprocess.tsv`) | 11.000 | 10.933 (67 duplikat dibuang) |
| Validasi (`valid_preprocess.tsv`) | 1.260 | ±1.246 (14 teks yang juga ada di data latih dibuang) |
| Uji (`test_preprocess_masked_label.tsv`) | 500 | Tidak dipakai, label disembunyikan |

Distribusi label latih timpang: positif 58%, negatif 31%, netral 10%.

Pembersihan dilakukan otomatis di notebook:
1. Buang duplikat persis (teks + label).
2. Buang teks yang memiliki label bertentangan.
3. Buang teks validasi yang juga muncul di data latih (mencegah kebocoran data).

## Protokol evaluasi

Label data uji resmi tidak tersedia, sehingga **validasi resmi yang sudah dibersihkan dipakai
sebagai held-out set**. Agar evaluasi tetap jujur:
- Tidak ada early stopping dan tidak ada pemilihan checkpoint berdasarkan validasi.
- Setiap run dilatih dengan jumlah epoch tetap, lalu model akhir dievaluasi satu kali.

## Backbone

| Kode | Checkpoint | Keterangan |
|---|---|---|
| `mbert` | `bert-base-multilingual-cased` | Baseline multibahasa |
| `distilbert` | `distilbert-base-multilingual-cased` | Versi ringkas (6 layer), untuk melihat efisiensi |
| `indobert` | `indobenchmark/indobert-base-p1` | Dilatih khusus Bahasa Indonesia (Indo4B) |
| `xlmr` | `xlm-roberta-base` | Multibahasa modern, hanya full fine-tuning |

## Strategi fine-tuning

Semua run memakai **head replacement**: classification head baru dengan 3 output.

| Strategi | Bagian yang dilatih | Learning rate |
|---|---|---|
| Full fine-tuning | Semua parameter | 2e-5 |
| Partial freezing | Embedding + separuh layer awal dibekukan; separuh layer akhir, pooler, dan head dilatih | 2e-5 |
| Linear probing | Seluruh backbone dibekukan; hanya head yang dilatih | 1e-3 |

## Matriks eksperimen (24 run)

| Backbone \ Strategi | Full | Partial | Probing |
|---|---|---|---|
| mBERT | ✓ ×3 seed | ✓ ×3 seed | — |
| DistilBERT | ✓ ×3 seed | ✓ ×3 seed | — |
| IndoBERT | ✓ ×3 seed | ✓ ×3 seed | ✓ ×3 seed |
| XLM-R | ✓ ×3 seed | — | — |

- Tiga backbone (mBERT, DistilBERT, IndoBERT) dibandingkan dengan protokol identik pada full dan partial.
- IndoBERT menjalankan ketiga strategi, sehingga pengaruh strategi bisa dilihat pada satu backbone.

## Variabel terkendali

| Variabel | Nilai |
|---|---|
| `max_length` | 128 token (p99 panjang kalimat = 79 kata) |
| Batch size latih | 32 |
| Epoch | 3 |
| Seed | 42, 123, 2024 |
| Optimizer | AdamW, weight decay 0,01 |
| Scheduler | Linear, warmup 10% dari total step |
| Presisi | fp16 (mixed precision) bila GPU tersedia |
| Data latih | Penuh (10.933 kalimat) |

## Metrik per run

- Performa: akurasi, F1 makro, F1 per kelas (negatif, netral, positif), confusion matrix.
- Efisiensi: waktu latih, waktu inferensi per kalimat (batch 1, 200 kalimat, 10 kalimat warmup),
  jumlah parameter total dan yang dilatih, memori GPU puncak.

## Hipotesis

- **H1 (backbone):** IndoBERT mencapai F1 makro tertinggi dibanding mBERT dan DistilBERT pada strategi
  yang sama, karena korpus pretraining dan kosakatanya selaras dengan Bahasa Indonesia. XLM-R diperkirakan
  bersaing dengan IndoBERT, dengan biaya parameter dan inferensi lebih besar.
- **H2 (strategi):** full fine-tuning > partial freezing > linear probing pada F1 makro. Partial freezing
  kehilangan sedikit F1 tetapi lebih hemat waktu latih dan memori; linear probing paling hemat tetapi
  F1-nya jauh lebih rendah.
- **H3 (efisiensi):** DistilBERT memiliki sekitar 40% lebih sedikit parameter dan inferensi sekitar
  1,6–2× lebih cepat dibanding mBERT, dengan penurunan F1 makro yang kecil (Sanh dkk., 2019).

## Aturan menyimpulkan

Dua konfigurasi dianggap **berbeda secara meyakinkan hanya jika selisih rata-rata F1 makro lebih
besar daripada simpangan baku antar-seed** (diambil yang terbesar dari keduanya).

## Platform

- Pengembangan dan uji cepat: Jupyter lokal (RTX 4050 Laptop 6 GB), `MODE="uji"`.
- Eksperimen resmi: Kaggle (GPU T4), `MODE="penuh"`, seluruh 24 run dalam satu sesi agar waktu
  inferensi dapat dibandingkan secara adil. Hasil lokal tidak dicampur dengan hasil Kaggle.

## Keterbatasan

- Evaluasi memakai validasi resmi karena label uji tidak tersedia.
- Hyperparameter tidak dituning per backbone; nilainya sama untuk semua agar perbandingan adil.
- Hanya satu dataset, sehingga kesimpulan belum tentu berlaku untuk tugas NLP lain.
- Waktu latih dan inferensi bergantung pada jenis GPU.