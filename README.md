# Pendo - Face Recognition Application

<div align="center">

![Pendo Logo](src/icons/generated_image.png)

*A modern, futuristic face recognition system with real-time scanning and beautiful UI*

</div>

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## 🎯 Overview

**Pendo** is a sophisticated face recognition application that combines cutting-edge AI technology with an elegant, futuristic user interface. Built with Python, it uses the EdgeFace model for accurate face recognition and features a real-time camera feed with smooth animations and intuitive controls.

### Key Technologies

- **Face Recognition**: EdgeFace model with YuNet face detection
- **Deep Learning**: PyTorch framework
- **Computer Vision**: OpenCV
- **UI Framework**: Tkinter with custom components
- **Image Processing**: PIL/Pillow

## ✨ Features

### Core Functionality

- **Real-time Face Recognition**: Instant identification of registered faces
- **Face Registration**: Easy enrollment of new faces with preview
- **Confidence Scoring**: Visual confidence indicators for matches
- **Multi-face Detection**: Detects and tracks multiple faces simultaneously
- **Adaptive Thresholds**: Configurable recognition sensitivity

### User Interface

- **Futuristic Design**: Glassmorphic cards with neon accents
- **Animated Background**: Dynamic network nodes and connections
- **Smooth Animations**: 60 FPS rendering with fluid transitions
- **Circular Scan Button**: Pulsating button with progress indicators
- **Side Panel**: Sliding recognition card with user information
- **Wave Effects**: Scanning animations overlay on camera feed

### Advanced Features

- **Dataset Management**: Organize face images by user
- **Model Training**: Train custom recognition models
- **Persistence**: Save and load face embeddings
- **Logging**: Comprehensive logging for debugging
- **Error Handling**: Robust error recovery

## 🏗️ Architecture

Pendo follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────┐
│           Application Layer                  │
│  (main_window.py - orchestrates components)  │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┴───────┐
       ↓               ↓
┌──────────────┐  ┌──────────────┐
│ UI Components│  │  Core Engine │
├──────────────┤  ├──────────────┤
│ • Background │  │ • Recognition│
│ • ScanButton │  │ • Training   │
│ • VideoFeed  │  │ • Dataset    │
│ • RecCard    │  │              │
└──────────────┘  └──────────────┘
       ↓               ↓
┌──────────────┐  ┌──────────────┐
│ UI Utilities │  │   Models     │
├──────────────┤  ├──────────────┤
│ • Colors     │  │ • EdgeFace   │
│ • IconLoader │  │ • YuNet      │
└──────────────┘  └──────────────┘
```

### Component Breakdown

#### UI Components (`src/ui/components/`)

- **background.py**: Animated gradient background with network nodes
- **scan_button.py**: Circular scan button with progress ring
- **video_feed.py**: Camera feed with recognition overlay
- **recognition_card.py**: Glassmorphic result card with actions

#### Core Engine (`src/core/`)

- **recognition.py**: Face detection and recognition logic
- **training.py**: Model training with triplet loss
- **dataset.py**: Dataset management and validation

#### Utilities (`src/ui/utils/`)

- **colors.py**: Centralized color palette
- **icon_loader.py**: Image loading and processing

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- macOS, Linux, or Windows

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/face_recognition_app.git
cd face_recognition_app
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements.txt contents:**
```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
```

### Step 4: Download Models

The application requires two models:

1. **EdgeFace Model**: Place in `edgeface/checkpoints/`
   - Download from: [EdgeFace Repository](https://github.com/edgeface/edgeface)
   - File: `edgeface_xs_gamma_06.pt`

2. **YuNet Face Detector**: Place in `models/`
   - Download from: [OpenCV Zoo](https://github.com/opencv/opencv_zoo)
   - File: `face_detection_yunet_2023mar.onnx`

```bash
# Create directories
mkdir -p edgeface/checkpoints
mkdir -p models

# Download models (example using curl)
curl -L <model_url> -o edgeface/checkpoints/edgeface_xs_gamma_06.pt
curl -L <detector_url> -o models/face_detection_yunet_2023mar.onnx
```

### Step 5: Verify Installation

```bash
python src/app.py
```

If everything is set up correctly, the Pendo window should open with your camera feed.

## 📖 Usage

### Starting the Application

```bash
python src/app.py
```

### Registering a Face

1. Click the **Scan Button** (top-right corner with Pendo icon)
2. Position your face in the camera view
3. Wait for the scan to complete (0-100%)
4. When a face is detected, the **Recognition Card** slides in from the right
5. Click the **SAVE** button (↓)
6. Enter your name in the dialog
7. Your face is now registered!

### Recognizing Faces

1. Click the **Scan Button** to start scanning
2. The camera will continuously scan for faces
3. When a registered face is detected:
   - The **Recognition Card** appears with the name
   - **Match Confidence** bar shows similarity percentage
   - Large percentage number displays confidence
4. Click **Scan Button** again to stop scanning

### Action Buttons

The Recognition Card features three action buttons:

- **↻ RESCAN**: Start a new scan session
- **↓ SAVE**: Register the current face with a name
- **→ SHARE**: (Coming soon) Share recognition results

### Menu

Click the **hamburger menu** (top-left) to access:
- Settings (coming soon)
- View registered users (coming soon)
- About

## 📁 Project Structure

```
face_recognition_app/
├── src/
│   ├── ui/
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── background.py          # Animated background
│   │   │   ├── scan_button.py         # Scan button component
│   │   │   ├── video_feed.py          # Camera feed
│   │   │   └── recognition_card.py    # Result card
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── colors.py              # Color constants
│   │   │   └── icon_loader.py         # Icon utilities
│   │   └── main_window.py             # Main UI controller
│   ├── core/
│   │   ├── recognition.py             # Face recognition engine
│   │   ├── training.py                # Model training
│   │   └── dataset.py                 # Dataset management
│   ├── icons/
│   │   └── generated_image.png        # Application icon
│   └── app.py                         # Application entry point
├── edgeface/
│   ├── backbones/                     # EdgeFace model architecture
│   └── checkpoints/
│       └── edgeface_xs_gamma_06.pt   # Pre-trained model
├── models/
│   └── face_detection_yunet_2023mar.onnx  # Face detector
├── dataset/                           # Face image dataset
│   └── [person_name]/                 # One folder per person
│       ├── 1.jpg
│       ├── 2.jpg
│       └── ...
├── embeddings/
│   └── face_embeddings.npy           # Saved face embeddings
├── config.py                          # Configuration settings
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

### File Descriptions

| File | Purpose | Lines | Complexity |
|------|---------|-------|------------|
| `main_window.py` | Main UI orchestration | ~260 | Low |
| `background.py` | Background animations | ~90 | Low |
| `scan_button.py` | Scan button logic | ~150 | Low |
| `video_feed.py` | Camera management | ~120 | Medium |
| `recognition_card.py` | Result display | ~280 | Medium |
| `recognition.py` | Face recognition | ~300 | High |
| `training.py` | Model training | ~200 | High |
| `dataset.py` | Dataset handling | ~150 | Medium |

## ⚙️ Configuration

Edit `config.py` to customize application behavior:

```python
# Model Configuration
EDGEFACE_MODEL_NAME = "edgeface_xs_gamma_06"
MODEL_INPUT_SIZE = (112, 112)

# Recognition Settings
RECOGNITION_THRESHOLD = 0.8  # Lower = more strict
CONFIDENCE_THRESHOLD = 0.5   # Minimum detection confidence

# Face Detection
MIN_FACE_SIZE = 80           # Minimum face size in pixels

# Paths
DATASET_DIR = "dataset"
EMBEDDINGS_FILE = "embeddings/face_embeddings.npy"
```

### Tuning Recognition Threshold

- **High threshold (0.9-1.0)**: More false positives, accepts similar faces
- **Medium threshold (0.7-0.9)**: Balanced accuracy (recommended)
- **Low threshold (0.5-0.7)**: More strict, fewer false positives

## 🛠️ Development

### Adding New Features

1. **UI Components**: Add to `src/ui/components/`
2. **Core Logic**: Add to `src/core/`
3. **Utilities**: Add to `src/ui/utils/` or `src/utils/`

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Keep functions under 50 lines
- Add docstrings to all public methods
- Use meaningful variable names

### Testing

```bash
# Test camera feed
python minimal_test.py

# Test recognition only
python test_camera.py
```

### Building Components

When creating new UI components:

1. Inherit from appropriate base class
2. Implement `draw()` method
3. Keep state management simple
4. Use color constants from `utils/colors.py`

Example:
```python
from ..utils.colors import CYAN, GREEN

class MyComponent:
    def __init__(self, canvas):
        self.canvas = canvas

    def draw(self):
        # Drawing logic here
        pass
```

## 🐛 Troubleshooting

### Camera Not Working

**Issue**: Black screen or "Cannot access camera"

**Solutions**:
- Check camera permissions in system settings
- Ensure no other app is using the camera
- Try different camera index: `VideoFeed(camera_index=1)`
- On macOS: Grant Terminal camera access

### ModuleNotFoundError

**Issue**: Missing Python packages

**Solution**:
```bash
pip install -r requirements.txt
```

### Model Not Found

**Issue**: "Model not found" error on startup

**Solution**:
- Verify model files are in correct locations:
  - `edgeface/checkpoints/edgeface_xs_gamma_06.pt`
  - `models/face_detection_yunet_2023mar.onnx`
- Check file permissions

### Low Recognition Accuracy

**Issue**: Faces not being recognized correctly

**Solutions**:
- Adjust `RECOGNITION_THRESHOLD` in `config.py`
- Ensure good lighting conditions
- Register multiple photos of each person
- Face should be frontal and clearly visible

### Slow Performance

**Issue**: Laggy UI or slow frame rate

**Solutions**:
- Reduce camera resolution in `video_feed.py`
- Lower animation frame rate (change `self.after(16, ...)` to higher value)
- Close other applications
- Use GPU if available (CUDA)

### Face Not Detected

**Issue**: "No face detected" when registering

**Solutions**:
- Ensure face is well-lit
- Move closer to camera
- Remove glasses/mask if applicable
- Check `MIN_FACE_SIZE` in config

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

<div align="center">

**Made with ❤️ using Python, PyTorch, and OpenCV**

[Report Bug](https://github.com/yourusername/face_recognition_app/issues) · [Request Feature](https://github.com/yourusername/face_recognition_app/issues)

</div>
