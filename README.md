# AI Vision Assistant Pro

A comprehensive AI-powered vision assistant with object detection, text recognition, and text-to-speech capabilities.

## 📁 Project Structure

```
project/
│
├── app.py                          # Main Flask application
├── templates/
│   └── index.html                  # HTML template
├── static/
│   ├── style.css                   # CSS styling
│   └── script.js                   # JavaScript functionality
├── audio_cache/                    # Auto-generated audio cache folder
└── README.md                       # This file
```

## 🚀 Features

### 1. **Object Detection Tab** 🎯
- Real-time object detection using YOLOv8
- Live camera feed with bounding boxes
- Audio announcements of detected objects
- Toggle detection on/off
- Voice commands support

### 2. **Text Recognition Tab** 📄
- OCR-based text detection from camera
- Real-time text reading aloud
- Optimized for document scanning
- Visual feedback with detection box

### 3. **Text Input Tab** 🔊 (NEW!)
- Manual text input interface
- Text-to-speech conversion
- Character counter
- Sample text buttons
- Keyboard shortcuts (Ctrl+Enter to speak)

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam/Camera
- Microphone (for voice commands)
- Tesseract OCR installed

## 🛠️ Installation

### Step 1: Install Tesseract OCR

**Windows:**
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to: `C:\Program Files\Tesseract-OCR\`
3. Add to PATH (optional)

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### Step 2: Install Python Dependencies

```bash
pip install flask opencv-python-headless ultralytics gtts pygame speechrecognition pytesseract pyttsx3 numpy pyaudio
```

**For Windows users, also install:**
```bash
pip install pywin32
```

### Step 3: Download YOLO Model

The YOLOv8s model will be downloaded automatically on first run, or download manually:
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
```

## 🎮 Running the Application

1. Navigate to project directory:
```bash
cd /path/to/project
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and go to:
```
http://localhost:5001
```

## 🎤 Voice Commands

- **"Open Camera"** - Start object detection
- **"Open Text Camera"** - Start text recognition
- **"Close"** - Stop any active camera
- **"Toggle Detection"** - Enable/disable object detection

## 💡 Usage Tips

### Object Detection:
- Ensure good lighting
- Keep objects within camera view
- Wait 5 seconds between audio announcements

### Text Recognition:
- Place document 8 inches from camera
- Center text in green detection box
- Use good lighting and contrast
- Hold document steady

### Text Input:
- Type or paste any text
- Click "Speak Text" or press Ctrl+Enter
- Use sample buttons for quick testing
- Clear text when done

## 🔧 Troubleshooting

### Camera not working:
- Check camera permissions
- Ensure camera is not in use by another application
- Try restarting the application

### Voice commands not responding:
- Check microphone permissions
- Speak clearly and wait for response
- Adjust microphone in system settings

### Text-to-speech not working:
- On Windows: Ensure SAPI voices are installed
- Check audio output device
- Try restarting the application

### Port 5001 already in use:
Change the port in `app.py`:
```python
app.run(debug=False, host='0.0.0.0', port=5002, threaded=True)
```

## 🎨 Customization

### Change Speech Rate:
In `app.py`, modify the TTS rate:
```python
engine.setProperty('rate', 150)  # Default is 150
```

### Change Detection Confidence:
In `app.py`, modify:
```python
CONFIDENCE_THRESHOLD = 0.6  # Range: 0.0 to 1.0
```

### Change Theme Colors:
Edit `static/style.css` to customize the color scheme.

## 📝 Technical Details

### Technologies Used:
- **Backend**: Flask (Python web framework)
- **Object Detection**: YOLOv8 (Ultralytics)
- **OCR**: Tesseract
- **TTS**: pyttsx3, gTTS, Windows SAPI
- **Speech Recognition**: Google Speech Recognition API
- **Frontend**: HTML5, CSS3, JavaScript

### System Requirements:
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Processor**: Dual-core 2.0GHz or better
- **Camera**: 720p or higher
- **OS**: Windows 10/11, Linux, macOS

## 🔒 Privacy & Security

- All processing is done locally
- No data is sent to external servers (except Google Speech API for voice commands)
- Camera feed is not recorded or stored
- Audio cache can be cleared manually from `audio_cache/` folder

## 📄 License

This project is for educational and personal use.

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure Tesseract is properly installed
4. Check console output for error messages

## 🎯 Future Enhancements

- [ ] Multi-language support
- [ ] Custom object training
- [ ] Speech rate adjustment in UI
- [ ] Dark mode theme
- [ ] Export detected text
- [ ] Recording history

---

**Enjoy using AI Vision Assistant Pro! 🚀**