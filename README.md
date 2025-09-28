# Face Recognition System - Python Implementation

A Python implementation of a face recognition system based on the Face-Recognition-Jetson-Nano project, with optional ArcFace / MobileFaceNet feature extraction and Z-score normalization.

---
## 🔰 Quick Start

### What you need to install (Apa saja yang perlu di‑install)
1. Python (recommended 3.11) 
2. Git (optional, if cloning)  
3. Project dependencies:
  ```bash
  pip install -r requirements.txt
  ```
4. (Optional – for better accuracy / performance):
  ```bash
  pip install onnxruntime  # ONNX ArcFace models
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # Optional
  ```
  Catatan: Paket `ncnn` sudah otomatis ada di `requirements.txt`. Jika instalasi gagal di platform Anda, hapus atau comment barisnya di `requirements.txt`; sistem akan fallback otomatis.

### Jalankan (Run)
Run dengan camera
```bash
python face_recognition_main.py 0 
```

Run dengan video
```bash
python face_recognition_main.py kdmjeje2.mp4
```

Run dengan citra
```bash
python face_recognition_main.py kdmjejetest.jpg
```


Populate database (opsional) sebelum jalan:
```bash
python tools/mobilefacenet.py --images images_processed --clear --populate --use-zscore-norm
```

Lalu jalankan dengan database itu:
```bash
python face_recognition_main.py 0 --database face_database_mobilefacenet.json --use-arcface --use-zscore-norm
```

Mode legacy tanpa ArcFace:
```bash
python face_recognition_main.py 0 --use-legacy --threshold 0.8
```

Auto tambah wajah baru saat runtime:
```bash
python face_recognition_main.py 0 --use-arcface --use-zscore-norm --auto-add
```

Tekan `s` untuk save database, `q` untuk keluar.

---
## 📌 Cara Menjalankan
1. Install requirements
2. (Opsional) Populate database
3. Jalankan aplikasi webcam
4. Tekan `q` untuk exit / `s` untuk save


