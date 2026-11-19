# Indo Sentiment Fine-tuning Benchmark

Proyek ini membandingkan beberapa model Transformer pretrained dan strategi fine-tuning
untuk klasifikasi sentimen Bahasa Indonesia (negatif, netral, positif). Fokusnya adalah
melihat seberapa besar pengaruh pilihan model dan jumlah parameter yang dilatih terhadap
performa dan efisiensi.

## Fokus:

- **Model:** IndoBERT, mBERT, DistilBERT multilingual, dan XLM-R.
- **Strategi fine-tuning:** full fine-tuning, partial freezing (separuh layer awal dibekukan),
  dan linear probing (hanya head yang dilatih).
- **Evaluasi:** F1 makro, akurasi, waktu latih, waktu inferensi, jumlah parameter, dan memori GPU.
  Setiap konfigurasi dijalankan dengan 3 seed untuk melihat konsistensi hasil.

## Dataset

[SmSA](https://github.com/IndoNLP/indonlu/tree/master/dataset/smsa_doc-sentiment-prosa)
dari benchmark IndoNLU (Wilie dkk., 2020). Dataset diunduh otomatis oleh notebook.

## Struktur

- `notebooks/`: notebook eksperimen
- `docs/`: rancangan eksperimen
- `results/`: hasil setiap run dan grafik
