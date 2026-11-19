# SIMUTEH - Praktikum P5-P6 Computer Vision
Nama: Hisham | NIM: 2330511007 | Kelas: 7A | SEED: 1007

## Cara menjalankan
1. `pip install -r requirements.txt`
2. Dari folder utama: `python src/buat_dataset_sintetis.py` (SEED = 1007)
3. Jalankan notebook berurutan dengan Restart & Run All: B1 sampai B5, `P5_tepi.ipynb`, lalu `P6_morfologi.ipynb`.

## Memakai analisis bercak
```python
import sys; sys.path.append("src")
from praproses import praproses
import bercak
hasil = bercak.analisis_daun(praproses(citra_bgr, param_praproses))
hasil["jumlah_bercak"], hasil["persen_bercak"], hasil["kelas"]
```
Parameter disimpan di `hasil/param_bercak.json`; praproses di `hasil/param_praproses.json`.

## Alur
praproses -> mask daun awal (Otsu kanal S) -> bercak awal (median + k x MAD, kanal R) -> opening dan closing (SE elips 5) -> komponen terhubung (konektivitas 8) -> saring luas, kebulatan, rasio aspek -> persen bercak dan kelas.

## Ringkasan evaluasi 60 citra (`hasil/p6_evaluasi_bercak.csv`)
```
                                          Nilai
Galat absolut rerata persen bercak        0.329
Korelasi persen prediksi dan acuan        0.973
Akurasi kelas                             0.933
Jumlah daun sehat yang dinilai berbercak      1
Jumlah daun berbercak yang dinilai sehat      2
Kondisi cahaya dengan akurasi terendah    gelap
```
Mask daun dan bercak awal bersifat sementara; diganti metode segmentasi yang lebih kuat pada P7.
