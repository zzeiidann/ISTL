# Daily Screener

Workflow lokal utama sekarang memakai model Kronos IDX TF15:

1. `01_update_compound_tf15.ipynb` mengunduh rolling data Yahoo Finance,
   membuang candle yang belum selesai, lalu menggabungkannya dengan parquet
   historis tanpa menghapus data lama. Cell terakhir commit dan push hanya file
   parquet canonical ke `origin/main`.
2. `02_project_next_session_tf15.ipynb` memakai context aktual paling baru dan
   memproyeksikan semua candle pada sesi bursa berikutnya. Ranking utamanya
   berdasarkan return candle pertama (09:00–09:15 WIB).
3. `03_run_tf15_on_kaggle_t4.ipynb` mengirim full-universe inference ke Kaggle
   T4, memantau status, dan mengunduh output kembali tanpa membuka Kaggle setiap run.

Notebook lama TF15-vs-TF1D dan TF30 tetap disimpan sebagai eksperimen historis;
dua notebook bernomor di atas adalah workflow produksi lokal.

## Menyiapkan environment dengan uv

Dari repository root:

```bash
cd "Daily Screener"
uv sync
uv run python -m ipykernel install --user --name daily-screener --display-name "Daily Screener (uv)"
uv run jupyter lab
```

Pilih kernel **Daily Screener (uv)**, jalankan notebook `01`, kemudian `02`.
Model TF15 dan parquet canonical sudah dibaca langsung dari repository lokal.
Source Kronos resmi akan di-clone sekali ke `.runtime/Kronos`, sedangkan tokenizer
akan diunduh dan disimpan dalam cache Hugging Face pada pemakaian pertama.

CPU didukung dan menjadi fallback otomatis. Seluruh ticker yang memiliki minimal
240 bar masuk inference tanpa pre-filter likuiditas. Konfigurasi default memakai
lima sampled paths dan batch dua pada CPU. Gunakan `SAMPLE_PATHS = 1` untuk smoke
test atau GPU CUDA untuk full-universe inference rutin. Output ranking lengkap,
Top 30, seluruh path, dan metadata ditulis ke `Daily Screener/outputs/` dan
sengaja tidak dilacak Git.

`TARGET_DATE = None` memilih weekday berikutnya. Karena itu tidak mengetahui
libur khusus BEI; isi tanggal secara manual pada notebook `02` bila sesi berikutnya
bukan weekday kalender terdekat.

## Menjalankan tanpa UI notebook

```bash
uv run python update_tf15_parquet.py
uv run python project_tf15_next_session.py
```

Notebook diregenerasi dari source dengan:

```bash
uv run python build_local_tf15_notebooks.py
```

## Login Kaggle satu kali

Kaggle CLI tersimpan di environment `uv`. Login OAuth sekali dengan:

```bash
cd "Daily Screener"
uv run kaggle auth login
```

Browser hanya diperlukan untuk menyetujui login pertama. Simpan username pada file
lokal yang sudah diabaikan Git:

```bash
KAGGLE_USERNAME=mraffy
```

di `Daily Screener/.env.kaggle.local`. Notebook `03` membacanya otomatis; submit
T4, polling status, dan download output tidak memerlukan username hard-coded.
Credential disimpan oleh Kaggle CLI di konfigurasi user dan tidak dimasukkan ke
repository. Alternatif resmi tanpa OAuth adalah menyimpan token dari Kaggle
Settings API ke `~/.kaggle/access_token`.

Runner T4 tidak mengunduh Yahoo lagi. Ia memakai parquet yang sudah di-update dan
di-push oleh notebook `01`, menarik checkpoint model LFS, lalu langsung inference.
