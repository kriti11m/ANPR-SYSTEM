# Vision Processing Module

Computer vision module for real-time license plate detection and recognition using YOLOv11 and OpenCV.

## 🏗️ Structure

```
vision/
├── core/              # Core processing modules
│   ├── detector.py    # YOLOv11 license plate detector
│   ├── recognizer.py  # OCR text recognition
│   └── processor.py   # Main processing pipeline
├── models/            # Pre-trained model files
│   ├── yolov11_anpr.pt
│   └── ocr_model.onnx
├── utils/             # Utility functions
│   ├── video_capture.py  # Video input handling
│   ├── image_utils.py    # Image processing utilities
│   └── visualization.py  # Result visualization
├── tests/             # Unit tests
└── requirements.txt   # Python dependencies
```

## 🚀 Features

- **Multiple Input Sources**: Webcam, RTSP streams, video files
- **YOLOv11 Integration**: State-of-the-art object detection
- **Real-time Processing**: Optimized for live video feeds
- **OCR Recognition**: Accurate license plate text extraction
- **Batch Processing**: Process multiple videos simultaneously
- **GPU Acceleration**: CUDA support for faster processing

## 🛠️ Setup

### Dependencies
```bash
cd vision

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
python scripts/download_models.py
```

### Requirements
- Python 3.8+
- OpenCV 4.5+
- PyTorch 1.12+
- Ultralytics YOLOv11
- TensorRT (optional, for GPU acceleration)

## 🎥 Video Capture Module

### Supported Input Sources

#### Webcam
```python
from vision.utils.video_capture import VideoCapture

# Default webcam (index 0)
capture = VideoCapture(source=0)

# USB camera (index 1)
capture = VideoCapture(source=1)
```

#### RTSP Stream
```python
# IP camera RTSP stream
rtsp_url = "rtsp://admin:password@192.168.1.100:554/stream"
capture = VideoCapture(source=rtsp_url)

# Multiple streams
streams = [
    "rtsp://camera1:554/stream1",
    "rtsp://camera2:554/stream2"
]
```

#### Video File
```python
# Local video file
capture = VideoCapture(source="path/to/video.mp4")

# Network video file
capture = VideoCapture(source="http://example.com/video.mp4")
```

### Continuous Frame Processing
```python
from vision.utils.video_capture import VideoCapture

def process_video_stream(source):
    """
    Continuously read frames from video source and yield them for processing.
    
    Args:
        source: Video source (int for webcam, str for file/stream)
    
    Yields:
        numpy.ndarray: Video frame
    """
    capture = VideoCapture(source)
    
    try:
        while True:
            frame = capture.read_frame()
            if frame is None:
                break
            
            # Yield frame for YOLO processing
            yield frame
            
    finally:
        capture.release()

# Usage example
for frame in process_video_stream(0):  # Webcam
    # Process frame with YOLO
    detections = detector.detect(frame)
    # Handle detections...
```

## 🧠 Detection Pipeline

### YOLOv11 License Plate Detector
```python
from vision.core.detector import LicensePlateDetector

detector = LicensePlateDetector(
    model_path="models/yolov11_anpr.pt",
    confidence_threshold=0.5,
    iou_threshold=0.4
)

# Detect license plates in frame
detections = detector.detect(frame)
```

### OCR Text Recognition
```python
from vision.core.recognizer import TextRecognizer

recognizer = TextRecognizer(
    model_path="models/ocr_model.onnx"
)

# Extract text from detected plate region
plate_text = recognizer.recognize(plate_region)
```

### Complete Processing Pipeline
```python
from vision.core.processor import ANPRProcessor

processor = ANPRProcessor(
    detector_model="models/yolov11_anpr.pt",
    recognizer_model="models/ocr_model.onnx"
)

# Process single frame
results = processor.process_frame(frame)

# Process video stream
for result in processor.process_stream(video_source):
    print(f"Detected: {result.plate_text} (confidence: {result.confidence})")
```

## 📊 Performance Optimization

### GPU Acceleration
```python
# Enable CUDA if available
detector = LicensePlateDetector(
    model_path="models/yolov11_anpr.pt",
    device="cuda" if torch.cuda.is_available() else "cpu"
)
```

### Batch Processing
```python
# Process multiple frames at once
frames_batch = [frame1, frame2, frame3, frame4]
results_batch = detector.detect_batch(frames_batch)
```

### Frame Skipping
```python
# Process every nth frame for better performance
processor = ANPRProcessor(frame_skip=3)  # Process every 3rd frame
```

## 🔧 Configuration

### Model Configuration
```yaml
detector:
  model_path: "models/yolov11_anpr.pt"
  confidence_threshold: 0.5
  iou_threshold: 0.4
  input_size: [640, 640]

recognizer:
  model_path: "models/ocr_model.onnx"
  max_text_length: 15
  character_set: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

video:
  fps: 30
  buffer_size: 5
  timeout: 30
```

### Performance Settings
```yaml
performance:
  batch_size: 4
  frame_skip: 1
  gpu_memory_fraction: 0.8
  num_workers: 4
```

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Test with sample video
python test_video.py --source samples/test_video.mp4

# Benchmark performance
python benchmark.py --model models/yolov11_anpr.pt
```

## 📈 Monitoring

### Metrics Collection
- Processing FPS
- Detection accuracy
- Recognition confidence scores
- GPU/CPU utilization
- Memory usage

### Logging
```python
import logging

logger = logging.getLogger("vision")
logger.setLevel(logging.INFO)

# Log detection results
logger.info(f"Plate detected: {plate_text}, confidence: {confidence}")
```

## 🔍 Debugging

### Visualization Tools
```python
from vision.utils.visualization import draw_detections

# Draw bounding boxes and text on frame
annotated_frame = draw_detections(frame, detections)

# Save debug images
cv2.imwrite("debug_output.jpg", annotated_frame)
```

### Performance Profiling
```python
import time
from vision.utils.profiler import profile_function

@profile_function
def process_frame(frame):
    return detector.detect(frame)
```

## 🚀 Deployment

### Docker Container
```dockerfile
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

### Production Considerations
- Model optimization (TensorRT, ONNX)
- Load balancing for multiple streams
- Caching frequently accessed frames
- Automatic model updates
- Error recovery and failover
