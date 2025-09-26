# Face Recognition System - Python Implementation

A Python implementation of face recognition system based on the Face-Recognition-Jetson-Nano project, but using Python APIs similar to jetson-inference.

## Features

- **Face Detection**: Uses OpenCV DNN or Haar cascades for face detection
- **Feature Extraction**: Implements ArcFace-like feature extraction (with fallback to simple features)
- **Face Database**: JSON-based face database with cosine similarity matching
- **Quality Assessment**: 
  - Blur detection using Laplacian variance
  - Liveness detection (anti-spoofing)
  - Face angle assessment
- **Real-time Processing**: Live camera or video file processing
- **Auto-populate**: Automatically add faces from image directories

## Project Structure

```
Face-Recognition-Python/
├── src/
│   ├── face_detector.py      # Face detection module
│   ├── face_features.py      # Feature extraction module  
│   ├── face_database.py      # Database management
│   └── face_quality.py       # Quality assessment
├── models/                   # Place model files here
├── images/                   # Training images directory
├── face_recognition_main.py  # Main application
├── config.json              # Configuration file
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Installation

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Models** (Optional):
   - For better face detection, download OpenCV face detection models
   - For better feature extraction, use ArcFace or similar models
   - Place model files in the `models/` directory

## Usage

### Basic Usage

Run with default webcam:
```bash
python face_recognition_main.py
```

Run with specific camera:
```bash
python face_recognition_main.py 1
```

Run with video file:
```bash
python face_recognition_main.py path/to/video.mp4
```

Run with image:
```bash
python face_recognition_main.py path/to/image.jpg
```

### Command Line Options

```bash
python face_recognition_main.py [input] [options]

Arguments:
  input                 Input source (camera index, image, or video file)

Options:
  --config CONFIG       Configuration file path
  --database DATABASE   Face database file path (default: face_database_mobilefacenet.json)
  --populate DIRECTORY  Auto-populate database from directory
  --threshold VALUE     Recognition threshold (auto-adjusted if omitted)
  --min-size SIZE       Minimum face size in pixels (default: 90)
  --show-legend         Show information legend (default)
  --hide-legend         Hide on-screen legend to reduce drawing cost
  --enable-liveness     Enable liveness detection (default)
  --disable-liveness    Disable liveness detection for extra speed
  --enable-blur-filter  Enable blur-based rejection (default)
  --disable-blur-filter Disable blur filter (slightly faster)
  --show-landmarks      Draw five-point facial landmarks
  --hide-landmarks      Skip landmark rendering (default in --fast mode)
  --detection-downscale FACTOR  Resize frames before detection (e.g. 0.6)
  --quality-interval N  Re-run blur/liveness every N frames (default: 1)
  --fast                Apply a tuned set of speed-oriented defaults
  --opencv-threads N    Override OpenCV thread count (0 lets OpenCV decide)
  --auto-add            Automatically add unknown faces when quality is good
```

### Examples

**Auto-populate database from images:**
```bash
python face_recognition_main.py --populate images/
```

**Run with custom settings:**
```bash
python face_recognition_main.py 0 --threshold 0.7 --min-size 100 --auto-add
```

**Process video file:**
```bash
python face_recognition_main.py video.mp4 --database my_faces.json
```

## Adding People to Database

### Method 1: Directory Structure
Create directories with person names and place their images inside:

```
images/
├── John_Doe/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── Jane_Smith/
│   ├── image1.png
│   └── image2.png
└── ...
```

Then run:
```bash
python face_recognition_main.py --populate images/
```

### Method 2: Auto-add during runtime
Run with `--auto-add` flag to automatically add unknown faces:
```bash
python face_recognition_main.py 0 --auto-add
```

### Method 3: Programmatically
```python
from src.face_database import FaceDatabase
from src.face_features import FaceFeatureExtractor

# Initialize components
extractor = FaceFeatureExtractor()
database = FaceDatabase()
database.set_feature_extractor(extractor)

# Add face from file
database.add_face_from_file("Person Name", "path/to/photo.jpg")

# Save database
database.save_database()
```

## Configuration

Create a `config.json` file to customize behavior:

```json
{
  "min_face_size": 90,
  "face_threshold": 0.5,
  "recognition_threshold": 0.6,
  "liveness_threshold": 0.93,
  "max_blur": -25.0,
  "max_angle": 10.0,
  "max_database_items": 2000,
  "show_landmarks": true,
  "show_legend": true,
  "enable_liveness": true,
  "enable_blur_filter": true,
  "auto_add_faces": false,
  "database_path": "face_database_mobilefacenet.json",
  "images_directory": "images",
  "detection_frame_size": [320, 240]
}
```

## Quality Assessment

The system includes several quality checks:

- **Blur Detection**: Rejects blurry faces using Laplacian variance
- **Liveness Detection**: Anti-spoofing using texture and color analysis
- **Face Angle**: Filters faces that are too tilted
- **Face Size**: Ensures faces are large enough for reliable recognition

## Performance

- **Real-time Processing**: Optimized for live camera streams
- **FPS Display**: Shows current processing speed
- **Quality vs Speed**: Configurable thresholds for performance tuning
- **Fast Mode**: Use `--fast` to automatically downscale detection, batch quality checks, and disable expensive filters for higher FPS
- **Custom Control**: Combine `--detection-downscale`, `--quality-interval`, `--disable-liveness`, and `--hide-landmarks` to tailor the speed/accuracy trade-off
- **Legacy Compatibility**: By default frames are resized to 320×240 before detection to mirror the original Jetson implementation. Set `"detection_frame_size": null` (or use a custom size) in your config to change this behavior.

## Controls

During runtime:
- **'q'**: Quit application
- **'s'**: Save database manually

## Comparison with Original C++ Version

| Feature | C++ Version | Python Version |
|---------|------------|----------------|
| Face Detection | MTCNN/RetinaFace | OpenCV DNN/Haar |
| Feature Extraction | ArcFace (NCNN) | Simple features/Custom |
| Database | Vector-based | JSON-based |
| Quality Assessment | ✓ | ✓ |
| Liveness Detection | ✓ | ✓ (simplified) |
| Real-time Processing | ✓ | ✓ |

## Extending the System

### Adding Better Models

1. **Face Detection**: Replace with MTCNN, RetinaFace, or similar
2. **Feature Extraction**: Use pre-trained ArcFace, FaceNet, or similar models
3. **Liveness Detection**: Integrate trained anti-spoofing models

### Custom Features

The modular design allows easy extension:
- Add new quality assessment methods
- Implement different similarity metrics  
- Create custom database backends
- Add new visualization options

## Troubleshooting

**Camera not working**:
- Try different camera indices (0, 1, 2, ...)
- Check camera permissions
- Ensure camera is not used by other applications

**Poor recognition accuracy**:
- Increase `--threshold` value
- Add more training images per person
- Ensure good lighting conditions
- Check image quality (not blurry, frontal face)

**Slow performance**:
- Reduce input resolution
- Increase `--min-size` to filter small faces
- Disable quality checks if not needed

## License

This project is based on the Face-Recognition-Jetson-Nano project. Please check the original project's license terms.

## Contributing

Feel free to submit issues, feature requests, and pull requests to improve the system.