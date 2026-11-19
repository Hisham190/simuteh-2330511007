# SIMUTEH - Evaluasi Praktikum P1-P4 Computer Vision
Nama: Hisham | NIM: 2330511007 | Kelas: 7A | SEED: 1007

## Cara menjalankan
1. `pip install -r requirements.txt`
2. Dari folder utama: `python src/buat_dataset_sintetis.py` (SEED = 1007)
3. Jalankan notebook B1 sampai B5 di folder `notebook/` berurutan dengan Restart & Run All.

## Rangkaian praproses final (`hasil/param_praproses.json`)
ubah ukuran 640 px -> gamma otomatis (terang 1,6; gelap dibiarkan) -> CLAHE kanal V (clip 1, tile 4, hanya bila cahaya tidak rata) -> median 3 x 3; tanpa penajaman.

## Ringkasan KB kanal R (rerata 40 citra berbercak)
```
                KB_sebelum  KB_sesudah  perubahan_%
kondisi                                            
normal               10.18       15.40         51.3
gelap                 9.11       16.09         76.5
terang                7.59       12.75         67.9
kontras_rendah        3.90        8.02        105.9
bayangan              4.14        6.91         66.8
```
