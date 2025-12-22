"""
YOLOv11 License Plate Detector

This module implements license plate detection using YOLOv11 from Ultralytics.
It processes video frames to detect license plates and returns bounding boxes
with cropped plate images for further OCR processing.
"""

import cv2
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import time

try:
    from ultralytics import YOLO
except ImportError:
    logging.error("Ultralytics not installed. Please install with: pip install ultralytics")
    raise

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class PlateDetection:
    """Data class for license plate detection results."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    cropped_plate: np.ndarray
    class_id: int
    timestamp: float


class YOLOv11PlateDetector:
    """
    YOLOv11-based license plate detector.
    
    This class handles:
    - Loading pretrained YOLOv11 models
    - Processing video frames for license plate detection
    - Extracting bounding boxes and cropped plate images
    - Performance optimization and batch processing
    """
    
    def __init__(
        self,
        model_path: str = "yolov11n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.4,
        device: str = "auto",
        input_size: int = 640
    ):
        """
        Initialize YOLOv11 license plate detector.
        
        Args:
            model_path: Path to YOLOv11 model file (.pt)
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
            iou_threshold: IoU threshold for NMS (0.0-1.0)
            device: Device to run inference ('cpu', 'cuda', 'auto')
            input_size: Input image size for YOLO (640, 1280, etc.)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.input_size = input_size
        
        # Performance tracking
        self.inference_times = []
        self.detections_count = 0
        self.frames_processed = 0
        
        # Load model
        self.model = self._load_model()
        
        logger.info(f"YOLOv11PlateDetector initialized:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Confidence threshold: {confidence_threshold}")
        logger.info(f"  IoU threshold: {iou_threshold}")
        logger.info(f"  Device: {self.model.device}")
        logger.info(f"  Input size: {input_size}")
    
    def _load_model(self) -> YOLO:
        """Load YOLOv11 model with error handling."""
        try:
            logger.info(f"Loading YOLOv11 model from: {self.model_path}")
            
            # Check if model file exists
            if not Path(self.model_path).exists() and not self.model_path.startswith('yolov11'):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load model
            model = YOLO(self.model_path)
            
            # Set device
            if self.device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            model.to(device)
            
            logger.info(f"Model loaded successfully on device: {device}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load YOLOv11 model: {e}")
            raise
    
    def detect_plates(
        self, 
        frame: np.ndarray,
        return_crops: bool = True
    ) -> List[PlateDetection]:
        """
        Detect license plates in a single frame.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            return_crops: Whether to return cropped plate images
        
        Returns:
            List[PlateDetection]: List of detected license plates
        """
        start_time = time.time()
        
        try:
            # Run YOLO inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.input_size,
                verbose=False
            )
            
            detections = []
            
            # Process results
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    
                    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                        # Convert coordinates to integers
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Validate bounding box
                        if not self._is_valid_bbox(x1, y1, x2, y2, frame.shape):
                            continue
                        
                        # Crop license plate region
                        cropped_plate = None
                        if return_crops:
                            cropped_plate = self._crop_plate(frame, x1, y1, x2, y2)
                        
                        # Create detection object
                        detection = PlateDetection(
                            bbox=(x1, y1, x2, y2),
                            confidence=float(conf),
                            cropped_plate=cropped_plate,
                            class_id=int(cls_id),
                            timestamp=time.time()
                        )
                        
                        detections.append(detection)
            
            # Update statistics
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.detections_count += len(detections)
            self.frames_processed += 1
            
            # Keep only recent inference times
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            
            logger.debug(f"Detected {len(detections)} plates in {inference_time:.3f}s")
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during plate detection: {e}")
            return []
    
    def detect_plates_batch(
        self, 
        frames: List[np.ndarray],
        return_crops: bool = True
    ) -> List[List[PlateDetection]]:
        """
        Detect license plates in a batch of frames (more efficient).
        
        Args:
            frames: List of input frames
            return_crops: Whether to return cropped plate images
        
        Returns:
            List[List[PlateDetection]]: List of detection lists for each frame
        """
        start_time = time.time()
        
        try:
            # Run batch inference
            results_list = self.model(
                frames,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.input_size,
                verbose=False
            )
            
            batch_detections = []
            
            # Process results for each frame
            for frame_idx, (frame, results) in enumerate(zip(frames, results_list)):
                frame_detections = []
                
                if results.boxes is not None:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    confidences = results.boxes.conf.cpu().numpy()
                    class_ids = results.boxes.cls.cpu().numpy()
                    
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        x1, y1, x2, y2 = map(int, box)
                        
                        if not self._is_valid_bbox(x1, y1, x2, y2, frame.shape):
                            continue
                        
                        cropped_plate = None
                        if return_crops:
                            cropped_plate = self._crop_plate(frame, x1, y1, x2, y2)
                        
                        detection = PlateDetection(
                            bbox=(x1, y1, x2, y2),
                            confidence=float(conf),
                            cropped_plate=cropped_plate,
                            class_id=int(cls_id),
                            timestamp=time.time()
                        )
                        
                        frame_detections.append(detection)
                
                batch_detections.append(frame_detections)
            
            # Update statistics
            inference_time = time.time() - start_time
            total_detections = sum(len(dets) for dets in batch_detections)
            
            self.inference_times.append(inference_time / len(frames))  # Average per frame
            self.detections_count += total_detections
            self.frames_processed += len(frames)
            
            logger.debug(f"Batch processed {len(frames)} frames, {total_detections} plates in {inference_time:.3f}s")
            
            return batch_detections
            
        except Exception as e:
            logger.error(f"Error during batch plate detection: {e}")
            return [[] for _ in frames]
    
    def _is_valid_bbox(self, x1: int, y1: int, x2: int, y2: int, frame_shape: Tuple[int, int, int]) -> bool:
        """Validate bounding box coordinates."""
        h, w = frame_shape[:2]
        
        # Check bounds
        if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h:
            return False
        
        # Check minimum size
        if (x2 - x1) < 20 or (y2 - y1) < 10:
            return False
        
        # Check aspect ratio (license plates are typically wider than tall)
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height
        
        if aspect_ratio < 1.5 or aspect_ratio > 6.0:  # Typical license plate aspect ratios
            return False
        
        return True
    
    def _crop_plate(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """
        Crop license plate region from frame with padding.
        
        Args:
            frame: Input frame
            x1, y1, x2, y2: Bounding box coordinates
        
        Returns:
            numpy.ndarray: Cropped plate image
        """
        h, w = frame.shape[:2]
        
        # Add padding around the detected region
        padding = 5
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(w, x2 + padding)
        y2_pad = min(h, y2 + padding)
        
        # Crop the region
        cropped = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        
        # Resize for consistent OCR processing (optional)
        # cropped = cv2.resize(cropped, (200, 50), interpolation=cv2.INTER_LINEAR)
        
        return cropped
    
    def visualize_detections(
        self, 
        frame: np.ndarray, 
        detections: List[PlateDetection],
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Draw detection results on frame for visualization.
        
        Args:
            frame: Input frame
            detections: List of plate detections
            show_confidence: Whether to show confidence scores
        
        Returns:
            numpy.ndarray: Frame with detection visualizations
        """
        viz_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            confidence = detection.confidence
            
            # Draw bounding box
            color = (0, 255, 0)  # Green for license plates
            thickness = 2
            cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw confidence score
            if show_confidence:
                label = f"Plate: {confidence:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 1
                
                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, font_thickness
                )
                
                # Draw background rectangle
                cv2.rectangle(
                    viz_frame,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    viz_frame,
                    label,
                    (x1, y1 - 5),
                    font,
                    font_scale,
                    (0, 0, 0),  # Black text
                    font_thickness
                )
        
        return viz_frame
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detector performance statistics."""
        if not self.inference_times:
            return {
                'frames_processed': self.frames_processed,
                'detections_count': self.detections_count,
                'avg_inference_time': 0,
                'avg_fps': 0,
                'detections_per_frame': 0
            }
        
        avg_inference_time = sum(self.inference_times) / len(self.inference_times)
        avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        detections_per_frame = self.detections_count / self.frames_processed if self.frames_processed > 0 else 0
        
        return {
            'frames_processed': self.frames_processed,
            'detections_count': self.detections_count,
            'avg_inference_time': avg_inference_time,
            'avg_fps': avg_fps,
            'detections_per_frame': detections_per_frame
        }
    
    def save_cropped_plates(
        self, 
        detections: List[PlateDetection], 
        output_dir: str = "detected_plates"
    ) -> List[str]:
        """
        Save cropped license plate images to disk.
        
        Args:
            detections: List of plate detections
            output_dir: Directory to save cropped plates
        
        Returns:
            List[str]: List of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_paths = []
        
        for i, detection in enumerate(detections):
            if detection.cropped_plate is not None:
                # Generate filename with timestamp
                timestamp = int(detection.timestamp * 1000)  # milliseconds
                filename = f"plate_{timestamp}_{i:03d}.jpg"
                file_path = output_path / filename
                
                # Save cropped plate
                cv2.imwrite(str(file_path), detection.cropped_plate)
                saved_paths.append(str(file_path))
                
                logger.debug(f"Saved cropped plate: {file_path}")
        
        return saved_paths


# Convenience functions for different model variants
def load_yolov11n_detector(**kwargs) -> YOLOv11PlateDetector:
    """Load YOLOv11 Nano model (fastest, least accurate)."""
    return YOLOv11PlateDetector(model_path="yolov11n.pt", **kwargs)


def load_yolov11s_detector(**kwargs) -> YOLOv11PlateDetector:
    """Load YOLOv11 Small model (balanced speed/accuracy)."""
    return YOLOv11PlateDetector(model_path="yolov11s.pt", **kwargs)


def load_yolov11m_detector(**kwargs) -> YOLOv11PlateDetector:
    """Load YOLOv11 Medium model (good accuracy)."""
    return YOLOv11PlateDetector(model_path="yolov11m.pt", **kwargs)


def load_yolov11l_detector(**kwargs) -> YOLOv11PlateDetector:
    """Load YOLOv11 Large model (high accuracy)."""
    return YOLOv11PlateDetector(model_path="yolov11l.pt", **kwargs)


def load_yolov11x_detector(**kwargs) -> YOLOv11PlateDetector:
    """Load YOLOv11 Extra Large model (highest accuracy, slowest)."""
    return YOLOv11PlateDetector(model_path="yolov11x.pt", **kwargs)


def load_custom_detector(model_path: str, **kwargs) -> YOLOv11PlateDetector:
    """Load custom trained YOLOv11 model."""
    return YOLOv11PlateDetector(model_path=model_path, **kwargs)


# Testing and demonstration functions
def test_detector_on_image(image_path: str, model_path: str = "yolov11n.pt"):
    """Test detector on a single image."""
    detector = YOLOv11PlateDetector(model_path=model_path)
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Detect plates
    detections = detector.detect_plates(frame)
    
    print(f"Found {len(detections)} license plates:")
    for i, detection in enumerate(detections):
        print(f"  Plate {i+1}: confidence={detection.confidence:.3f}, bbox={detection.bbox}")
    
    # Visualize results
    viz_frame = detector.visualize_detections(frame, detections)
    
    # Display result
    cv2.imshow("License Plate Detection", viz_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save cropped plates
    saved_paths = detector.save_cropped_plates(detections)
    print(f"Saved {len(saved_paths)} cropped plates")


def test_detector_on_webcam(device_id: int = 0, model_path: str = "yolov11n.pt"):
    """Test detector on webcam feed."""
    detector = YOLOv11PlateDetector(model_path=model_path)
    cap = cv2.VideoCapture(device_id)
    
    print("Testing license plate detection on webcam...")
    print("Press 'q' to quit, 's' to save detected plates")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect plates
        detections = detector.detect_plates(frame)
        
        # Visualize
        viz_frame = detector.visualize_detections(frame, detections)
        
        # Add stats overlay
        stats = detector.get_performance_stats()
        cv2.putText(
            viz_frame,
            f"FPS: {stats['avg_fps']:.1f} | Plates: {len(detections)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        cv2.imshow("License Plate Detection", viz_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and detections:
            saved_paths = detector.save_cropped_plates(detections)
            print(f"Saved {len(saved_paths)} plates")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final stats
    final_stats = detector.get_performance_stats()
    print(f"Final stats: {final_stats}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python detector.py image <path>    # Test on image")
        print("  python detector.py webcam [id]     # Test on webcam")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "image":
        if len(sys.argv) < 3:
            print("Please provide image path")
            sys.exit(1)
        test_detector_on_image(sys.argv[2])
    
    elif command == "webcam":
        device_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        test_detector_on_webcam(device_id)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
