# Screening abnormal price-volume IDX

Notebook utama: `screening_uma_idx.ipynb`.

## Input

- Universe dibaca dari `../Analisis Pendahuluan/KODE.xlsx` (959 emiten).
- Harga/volume harian diambil dari Yahoo Finance (`.JK`) dan disimpan sebagai
  cache Parquet di `data/raw/`.
- `config/control_universe.csv` harus diisi dengan konstituen IDX80/LQ45.
  Saham kontrol tetap masuk training dan scoring, tetapi otomatis dikeluarkan
  dari kandidat final.
- Jika riwayat UMA resmi belum tersedia, notebook otomatis memakai proxy
  suspected suspension dari minimal tiga sesi IHSG tanpa bar atau bervolume
  nol, yang diapit sesi bervolume positif. Proxy ini bukan bukti UMA/suspend
  resmi.

Contoh baris kontrol:

```csv
ticker,index_name,effective_from,effective_to
BBCA,IDX80,2025-02-01,2025-07-31
BBCA,LQ45,2025-02-01,2025-07-31
```

Input UMA resmi tetap disediakan bila kelak tersedia:

```csv
ticker,announcement_date,source_url
ABCD,2025-03-10,https://...
```

Tanggal efektif mencegah *look-ahead bias*. Jika riwayat komposisi belum
tersedia, isi snapshot terkini dan pahami bahwa filter kandidat final menjadi
aproksimasi.

## Menjalankan

```bash
python3 -m pip install -r "Model Screening UMA/requirements.txt"
jupyter lab "Model Screening UMA/screening_uma_idx.ipynb"
```

Jalankan sel secara berurutan. Optimasi dibatasi ke data sebelum 2026; periode
2026 hanya dibuka sekali untuk final test setelah bobot terkunci.

Definisi target riset pada notebook bersifat eksplisit dan dapat diubah:
kejadian positif bila dalam 20 hari bursa ke depan terdapat abnormal return
terhadap IHSG dan abnormal volume secara bersamaan. Ambang default tercantum
di bagian konfigurasi.
