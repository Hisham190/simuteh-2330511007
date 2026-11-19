"""praproses.py: rangkaian praproses SIMUTEH (dipakai ulang mulai P5)."""
import json
import cv2
import numpy as np

PARAM_BAWAAN = {
    "lebar": 640,             # sampling: lebar citra kerja (piksel)
    "gamma": "auto",          # angka tetap, atau "auto" (dipilih dari statistik kanal V)
    "gamma_terang": 1.6,      # dipakai bila gamma = "auto" dan citra terang
    "gamma_gelap": 1.0,       # dipakai bila gamma = "auto" dan citra gelap (1.0 = dibiarkan)
    "clahe_clip": 1.0,        # 0 berarti CLAHE tidak dipakai
    "clahe_tile": 4,
    "clahe_otomatis": True,   # True: CLAHE hanya bila cahaya terdeteksi tidak merata
    "batas_tidak_rata": 0.65, # ambang indeks ketidakrataan cahaya
    "penapis": "median",      # "none", "rerata", "gaussian", "median"
    "kernel": 3,
    "sigma": 1.0,
    "tajam_k": 0.0,           # 0 berarti tanpa penajaman
}


def _kanal_v(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]


def gamma_otomatis(img_bgr, gamma_terang=1.6, gamma_gelap=1.0):
    """Memilih gamma dari statistik kanal V. Ambang diturunkan dari statistik 60 citra
    sintetis: terang (rerata V > 210) -> gamma_terang; kontras rendah (simp. baku V < 20)
    -> 1.0; gelap (persentil 95 V < 200) -> gamma_gelap; selain itu -> 1.0."""
    v = _kanal_v(img_bgr)
    if v.mean() > 210:
        return gamma_terang
    if v.std() < 20:
        return 1.0
    if np.percentile(v, 95) < 200:
        return gamma_gelap
    return 1.0


def indeks_ketidakrataan(img_bgr):
    """Ketidakrataan cahaya: citra dibagi 4 x 4 blok, kecerahan tiap blok diwakili
    persentil 90 kanal V (didominasi latar nampan), lalu (maks - min) / rerata."""
    v = _kanal_v(img_bgr).astype(float)
    h, w = v.shape
    blok = [np.percentile(v[i * h // 4:(i + 1) * h // 4, j * w // 4:(j + 1) * w // 4], 90)
            for i in range(4) for j in range(4)]
    return (max(blok) - min(blok)) / np.mean(blok)


def praproses(img_bgr, p=None):
    """Urutan: ubah ukuran -> gamma -> CLAHE kanal V -> redam derau -> tajamkan."""
    p = {**PARAM_BAWAAN, **(p or {})}

    # 1. ubah ukuran, rasio aspek tetap
    h, w = img_bgr.shape[:2]
    img = cv2.resize(img_bgr, (p["lebar"], int(round(h * p["lebar"] / w))),
                     interpolation=cv2.INTER_AREA)

    # 2. koreksi gamma dengan lookup table
    g = (gamma_otomatis(img, p["gamma_terang"], p["gamma_gelap"])
         if p["gamma"] == "auto" else float(p["gamma"]))
    if g != 1.0:
        lut = np.clip(255 * (np.arange(256) / 255) ** g, 0, 255).astype(np.uint8)
        img = cv2.LUT(img, lut)

    # 3. CLAHE hanya pada kanal V agar warna tidak bergeser
    pakai = p["clahe_clip"] > 0
    if pakai and p["clahe_otomatis"]:
        pakai = indeks_ketidakrataan(img) > p["batas_tidak_rata"]
    if pakai:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=p["clahe_clip"],
                                tileGridSize=(p["clahe_tile"], p["clahe_tile"]))
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 4. redam derau
    k = p["kernel"]
    if p["penapis"] == "rerata":
        img = cv2.blur(img, (k, k))
    elif p["penapis"] == "gaussian":
        img = cv2.GaussianBlur(img, (k, k), p["sigma"])
    elif p["penapis"] == "median":
        img = cv2.medianBlur(img, k)

    # 5. tajamkan (unsharp masking) bila diminta
    if p["tajam_k"] > 0:
        blur = cv2.GaussianBlur(img, (5, 5), 1.0)
        img = cv2.addWeighted(img, 1 + p["tajam_k"], blur, -p["tajam_k"], 0)
    return img


def simpan_param(p, path="../hasil/param_praproses.json"):
    with open(path, "w") as f:
        json.dump({**PARAM_BAWAAN, **p}, f, indent=2)
