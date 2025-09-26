# Face Recognition System - Python Implementation

A Python implementation of face recognition system based on the Face-Recognition-Jetson-Nano project, following the API patterns from jetson-inference.

## 🚀 Features

- **Real-time face detection** using OpenCV DNN or Haar Cascades
- **Face feature extraction** with fallback to simple histogram-based features
- **Face database management** with JSON storage and cosine similarity matching
- **Quality assessment** including blur detection and liveness detection
- **Configurable system** with JSON-based configuration
- **Auto-population** from image directories
- **Live video processing** from webcam or video files

## 📋 Requirements

- **Python**: 3.8, 3.9, 3.10, or 3.11
- **Operating System**: Windows, Linux, macOS
- **Camera**: Optional (for live recognition)

## 🛠️ Installation

### 1. Clone or Download
```bash
cd Face-Recognition-Python
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python test_system.py
```

## 📁 Project Structure

```
Face-Recognition-Python/
├── src/                      # Core modules
│   ├── face_detector.py      # Face detection
│   ├── face_features.py      # Feature extraction
│   ├── face_database.py      # Database management
│   ├── face_quality.py       # Quality assessment
│   ├── config_manager.py     # Configuration
│   └── __init__.py
├── models/                   # AI models (optional)
├── images/                   # Training images
├── face_recognition_main.py  # Main application
├── test_system.py           # System tests
├── config.json              # Configuration file
└── requirements.txt         # Python dependencies
```

## 🎯 Quick Start

### Test the System
```bash
python test_system.py
```

### Run with Webcam
```bash
python face_recognition_main.py 0
```

### Process Video File
```bash
python face_recognition_main.py video.mp4
```

### Process Image
```bash
python face_recognition_main.py image.jpg
```

### Auto-populate Database
```bash
# Create directory structure: images/person_name/photo.jpg
mkdir -p images/john_doe
mkdir -p images/jane_smith

# Add photos to respective folders, then:
python face_recognition_main.py --populate images/
```

## ⚙️ Configuration

Edit `config.json` to customize behavior:

```json
{
  "min_face_size": 90,
  "recognition_threshold": 0.6,
  "liveness_threshold": 0.93,
  "show_landmarks": true,
  "enable_liveness": true,
  "auto_add_faces": false
}
```

## 🔧 Command Line Options

```bash
python face_recognition_main.py [input] [options]

Arguments:
  input                 Input source (0 for webcam, or file path)

Options:
  --config FILE         Configuration file path
  --database FILE       Database file path  
  --populate DIR        Auto-populate from directory
  --threshold FLOAT     Recognition threshold (0.0-1.0)
  --min-size INT        Minimum face size in pixels
  --show-legend         Show information overlay
  --enable-liveness     Enable liveness detection
  --auto-add            Auto-add unknown faces
```

## 📊 Performance Tips

### For Better Performance:
1. **Use smaller input resolution** for real-time processing
2. **Adjust thresholds** based on your use case
3. **Limit database size** for faster matching
4. **Use SSD storage** for database files

### For Better Accuracy:
1. **Add multiple photos** per person to database
2. **Use good quality images** (well-lit, frontal faces)
3. **Tune recognition threshold** based on testing
4. **Enable quality filters** (blur, liveness)

## 🔄 Upgrading to Advanced Models

### Optional Enhanced Models:

1. **Install additional packages:**
```bash
pip install dlib face-recognition mtcnn
```

2. **Download OpenCV DNN models:**
```bash
# Create models directory
mkdir models

# Download OpenCV face detector (optional)
wget https://github.com/opencv/opencv_3rdparty/raw/19512576c112aa2c7b6328cb0e8d589a4a90a26d/opencv_face_detector_uint8.pb -P models/
wget https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/opencv_face_detector.pbtxt -P models/
```

3. **Update config.json** to point to model files

## 🐛 Troubleshooting

### Common Issues:

**1. "No module named 'cv2'"**
```bash
pip install opencv-python
```

**2. "Could not find a version that satisfies the requirement"**
- Check your Python version with `python --version`
- Use Python 3.8-3.11
- Try creating a new virtual environment

**3. Camera not working**
- Check camera permissions
- Try different camera indices (0, 1, 2, ...)
- Test with: `python face_recognition_main.py 0`

**4. Poor recognition accuracy**
- Add more training photos per person
- Adjust `recognition_threshold` in config
- Ensure good image quality (lighting, resolution)

**5. Slow performance**
- Reduce input resolution
- Disable quality checks if not needed
- Limit database size

## 📈 System Architecture

```
Input (Camera/Video/Image)
    ↓
Face Detection (OpenCV DNN/Haar)
    ↓
Quality Assessment (Blur/Liveness/Angle)
    ↓
Feature Extraction (Simple/Custom Model)
    ↓
Database Matching (Cosine Similarity)
    ↓
Result Display (Labels/Statistics)
```

## 🤝 Contributing

Feel free to contribute improvements:

1. **Face Detection**: Add MTCNN or RetinaFace support
2. **Feature Extraction**: Integrate ArcFace or other models
3. **Database**: Add vector database support (Faiss, Pinecone)
4. **Quality**: Improve liveness detection algorithms
5. **Performance**: GPU acceleration, multi-threading

## 📄 License

This project is based on the Face-Recognition-Jetson-Nano project and follows similar open-source principles.

## 🆘 Support

- **Test first**: Run `python test_system.py`
- **Check logs**: Look for error messages in terminal
- **Verify config**: Ensure config.json is valid
- **Check dependencies**: Reinstall packages if needed

---

**Happy Face Recognition!** 🎉