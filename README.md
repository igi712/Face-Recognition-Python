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
python rebuild_database_zscore.py
```

Lalu jalankan dengan database itu:
```bash
python face_recognition_main.py 0 --database face_database.json --use-arcface --use-zscore-norm
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

---
## 📁 Output dan File yang Dihasilkan
Sistem ini menghasilkan beberapa output selama proses:

### 1. Dataset Wajah (dari `image_processor.py`)
- **Input**: Folder `images/` berisi gambar mentah
- **Output**: Folder `images_processed/` berisi gambar wajah yang sudah di-crop dan di-align (112x112 pixels)
- **Format**: JPG/PNG, satu folder per orang

### 2. Database Wajah (dari `rebuild_database_zscore.py`)
- **Input**: Folder `images_processed/`
- **Output**: File `face_database.json` berisi fitur wajah yang diekstrak dengan Z-score normalization
- **Backup**: Otomatis dibuat sebagai `face_database.json.backup_[timestamp]` sebelum rebuild

### 3. Recognition Real-time (dari `face_recognition_main.py`)
- **Input**: Webcam/video/citra
- **Output di Layar**: 
  - Bounding box wajah dengan nama/label
  - Confidence score
  - Landmark points (opsional)
  - Status liveness/blur detection (jika aktif)
- **Output File** (opsional):
  - `output/events.json`: Log event recognition (misal face_lost, face_detected) untuk robot greeting
  - Screenshot otomatis jika ada event tertentu

### 4. Folder Struktur
```
Face-Recognition-Python/
├── images/              # Input gambar mentah
├── images_processed/    # Output dataset wajah
├── face_database.json   # Database fitur wajah
├── output/              # Log events dan output lainnya
└── temp/                # File temporary
```

---
## ⚙️ Konfigurasi
- Edit `config.json` untuk setting default
- Argumen command-line override config
- Model ArcFace/ONNX untuk akurasi tinggi
- Z-score normalization untuk kompatibilitas Jetson Nano


