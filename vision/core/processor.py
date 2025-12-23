"""
ANPR Processing Pipeline

Main processing module that integrates video capture with YOLO detection
and provides a complete pipeline for real-time license plate recognition.
"""

import cv2
import logging
import time
from typing import Iterator, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from ..utils.video_capture import VideoCapture, MultiStreamCapture
from .detector import YOLOv11PlateDetector, PlateDetection

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Data class for ANPR detection results."""
    license_plate: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    timestamp: float
    frame_id: int
    source_id: str
    cropped_plate: Optional[np.ndarray] = None


class FrameProcessor:
    """
    Main processor for continuous frame analysis with YOLOv11 integration.
    
    This class provides the core functionality for processing video frames
    from multiple sources and detecting license plates using YOLOv11.
    """
    
    def __init__(
        self,
        target_fps: float = 30.0,
        frame_skip: int = 1,
        resize_dims: Optional[Tuple[int, int]] = None,
        yolo_model_path: str = "yolov11n.pt",
        confidence_threshold: float = 0.5,
        enable_yolo: bool = True,
        enable_ocr: bool = True
    ):
        """
        Initialize the frame processor with YOLOv11 integration and OCR.
        
        Args:
            target_fps: Target processing FPS
            frame_skip: Process every nth frame (1 = process all frames)
            resize_dims: Resize frames to (width, height) for processing
            yolo_model_path: Path to YOLOv11 model file
            confidence_threshold: Minimum confidence for detections
            enable_yolo: Whether to enable YOLO detection
            enable_ocr: Whether to enable OCR text extraction
        """
        self.target_fps = target_fps
        self.frame_skip = frame_skip
        self.resize_dims = resize_dims
        self.enable_yolo = enable_yolo
        self.enable_ocr = enable_ocr
        
        # Initialize YOLO detector if enabled
        self.detector = None
        if enable_yolo:
            try:
                self.detector = YOLOv11PlateDetector(
                    model_path=yolo_model_path,
                    confidence_threshold=confidence_threshold,
                    enable_ocr=enable_ocr
                )
                logger.info(f"YOLOv11 detector initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize YOLO detector: {e}")
                self.enable_yolo = False
        
        # Processing statistics
        self.frames_processed = 0
        self.processing_times = []
        self.last_process_time = 0
        self.total_detections = 0
        
        logger.info(f"FrameProcessor initialized:")
        logger.info(f"  Target FPS: {target_fps}")
        logger.info(f"  Frame skip: {frame_skip}")
        logger.info(f"  Resize dimensions: {resize_dims}")
        logger.info(f"  YOLO enabled: {enable_yolo}")
        if enable_yolo and self.detector:
            logger.info(f"  YOLO model: {yolo_model_path}")
            logger.info(f"  Confidence threshold: {confidence_threshold}")
    
    def process_single_source(
        self,
        source: str | int,
        source_id: str = "default"
    ) -> Iterator[Tuple[np.ndarray, Dict[str, Any], List[Detection]]]:
        """
        Process frames from a single video source with YOLO detection.
        
        Args:
            source: Video source (webcam index, file path, or RTSP URL)
            source_id: Identifier for this source
        
        Yields:
            Tuple[np.ndarray, Dict[str, Any], List[Detection]]: 
            (frame, metadata, detections)
        """
        logger.info(f"Starting processing for source: {source}")
        
        with VideoCapture(source) as capture:
            frame_counter = 0
            
            for frame in capture.get_frame_stream():
                frame_counter += 1
                
                # Skip frames if configured
                if frame_counter % self.frame_skip != 0:
                    continue
                
                # Preprocess frame
                processed_frame = self._preprocess_frame(frame)
                
                # Run YOLO detection
                detections = []
                if self.enable_yolo and self.detector:
                    detections = self._run_yolo_detection(
                        processed_frame, source_id, frame_counter
                    )
                
                # Prepare metadata
                metadata = {
                    'source_id': source_id,
                    'frame_id': frame_counter,
                    'timestamp': time.time(),
                    'original_shape': frame.shape,
                    'processed_shape': processed_frame.shape,
                    'capture_properties': capture.get_properties(),
                    'detections_count': len(detections)
                }
                
                # Update processing statistics
                self._update_processing_stats()
                self.total_detections += len(detections)
                
                yield processed_frame, metadata, detections
                
                # Control processing rate
                self._control_processing_rate()
    
    def process_multiple_sources(
        self,
        sources: List[Tuple[str | int, str]]
    ) -> Iterator[Tuple[np.ndarray, Dict[str, Any], List[Detection]]]:
        """
        Process frames from multiple video sources simultaneously with YOLO.
        
        Args:
            sources: List of (source, source_id) tuples
        
        Yields:
            Tuple[np.ndarray, Dict[str, Any], List[Detection]]: 
            (frame, metadata, detections)
        """
        logger.info(f"Starting processing for {len(sources)} sources")
        
        # Extract sources and IDs
        source_list = [source for source, _ in sources]
        source_ids = [source_id for _, source_id in sources]
        
        multi_capture = MultiStreamCapture(source_list)
        
        if not multi_capture.start_all():
            logger.error("Failed to start all captures")
            return
        
        try:
            frame_counters = {sid: 0 for sid in source_ids}
            
            while True:
                # Read frames from all sources
                frames_dict = multi_capture.read_all_frames()
                
                if not frames_dict:
                    time.sleep(0.01)  # Brief pause if no frames
                    continue
                
                # Process each available frame
                for stream_id, frame in frames_dict.items():
                    # Get corresponding source_id
                    stream_index = int(stream_id.split('_')[1])
                    source_id = source_ids[stream_index]
                    
                    frame_counters[source_id] += 1
                    
                    # Skip frames if configured
                    if frame_counters[source_id] % self.frame_skip != 0:
                        continue
                    
                    # Preprocess frame
                    processed_frame = self._preprocess_frame(frame)
                    
                    # Run YOLO detection
                    detections = []
                    if self.enable_yolo and self.detector:
                        detections = self._run_yolo_detection(
                            processed_frame, source_id, frame_counters[source_id]
                        )
                    
                    # Prepare metadata
                    metadata = {
                        'source_id': source_id,
                        'frame_id': frame_counters[source_id],
                        'timestamp': time.time(),
                        'original_shape': frame.shape,
                        'processed_shape': processed_frame.shape,
                        'capture_properties': multi_capture.get_all_properties().get(stream_id, {}),
                        'detections_count': len(detections)
                    }
                    
                    # Update processing statistics
                    self._update_processing_stats()
                    self.total_detections += len(detections)
                    
                    yield processed_frame, metadata, detections
                
                # Control processing rate
                self._control_processing_rate()
        
        finally:
            multi_capture.stop_all()
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame before YOLO detection.
        
        Args:
            frame: Original frame
        
        Returns:
            numpy.ndarray: Preprocessed frame
        """
        processed_frame = frame.copy()
        
        # Resize if specified
        if self.resize_dims:
            processed_frame = cv2.resize(
                processed_frame,
                self.resize_dims,
                interpolation=cv2.INTER_LINEAR
            )
        
        # Additional preprocessing can be added here
        # - Contrast/brightness adjustment
        # - Noise reduction
        # - Color space conversion
        
        return processed_frame
    
    def _run_yolo_detection(
        self, 
        frame: np.ndarray, 
        source_id: str, 
        frame_id: int
    ) -> List[Detection]:
        """
        Run YOLOv11 detection on frame and convert to Detection objects.
        
        Args:
            frame: Input frame for detection
            source_id: Source identifier
            frame_id: Frame number
        
        Returns:
            List[Detection]: List of license plate detections
        """
        try:
            # Run YOLO detection
            plate_detections = self.detector.detect_plates(frame, return_crops=True)
            
            # Convert to our Detection format
            detections = []
            current_time = time.time()
            
            for plate_det in plate_detections:
                # For now, set license_plate as placeholder since we don't have OCR yet
                # TODO: Integrate OCR to read actual license plate text
                license_plate = f"DETECTED_{frame_id}_{len(detections)}"
                
                detection = Detection(
                    license_plate=license_plate,
                    confidence=plate_det.confidence,
                    bbox=plate_det.bbox,
                    timestamp=current_time,
                    frame_id=frame_id,
                    source_id=source_id,
                    cropped_plate=plate_det.cropped_plate
                )
                
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during YOLO detection: {e}")
            return []
    
    def visualize_detections(
        self, 
        frame: np.ndarray, 
        detections: List[Detection],
        show_confidence: bool = True,
        show_text: bool = True
    ) -> np.ndarray:
        """
        Draw detection results on frame for visualization.
        
        Args:
            frame: Input frame
            detections: List of detections
            show_confidence: Whether to show confidence scores
            show_text: Whether to show license plate text
        
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
            
            # Prepare label text
            label_parts = []
            if show_text:
                label_parts.append(detection.license_plate)
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")
            
            if label_parts:
                label = " | ".join(label_parts)
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
    
    def _update_processing_stats(self):
        """Update processing statistics."""
        current_time = time.time()
        
        if self.last_process_time > 0:
            processing_time = current_time - self.last_process_time
            self.processing_times.append(processing_time)
            
            # Keep only recent processing times
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
        
        self.last_process_time = current_time
        self.frames_processed += 1
    
    def _control_processing_rate(self):
        """Control processing rate to match target FPS."""
        if self.target_fps <= 0:
            return
        
        target_interval = 1.0 / self.target_fps
        
        if self.processing_times:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)
            sleep_time = max(0, target_interval - avg_processing_time)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics including YOLO performance."""
        base_stats = {
            'frames_processed': self.frames_processed,
            'avg_processing_time': 0,
            'actual_fps': 0,
            'target_fps': self.target_fps,
            'frame_skip': self.frame_skip,
            'total_detections': self.total_detections,
            'detections_per_frame': 0,
            'yolo_enabled': self.enable_yolo
        }
        
        if not self.processing_times:
            return base_stats
        
        avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        actual_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        detections_per_frame = self.total_detections / self.frames_processed if self.frames_processed > 0 else 0
        
        base_stats.update({
            'avg_processing_time': avg_processing_time,
            'actual_fps': actual_fps,
            'detections_per_frame': detections_per_frame
        })
        
        # Add YOLO-specific stats if available
        if self.enable_yolo and self.detector:
            yolo_stats = self.detector.get_performance_stats()
            base_stats['yolo_stats'] = yolo_stats
        
        return base_stats


def process_webcam_stream(
    device_id: int = 0, 
    target_fps: float = 30.0,
    yolo_model: str = "yolov11n.pt",
    confidence: float = 0.5,
    enable_ocr: bool = True
) -> Iterator[Tuple[np.ndarray, Dict[str, Any], List[Detection]]]:
    """
    Convenience function for processing webcam stream with YOLO detection and OCR.
    
    Args:
        device_id: Webcam device ID
        target_fps: Target processing FPS
        yolo_model: YOLO model to use
        confidence: Confidence threshold
        enable_ocr: Whether to perform OCR on detected plates
    
    Yields:
        Tuple[np.ndarray, Dict[str, Any], List[Detection]]: 
        (frame, metadata, detections with OCR text)
    """
    processor = FrameProcessor(
        target_fps=target_fps,
        yolo_model_path=yolo_model,
        confidence_threshold=confidence,
        enable_ocr=enable_ocr
    )
    yield from processor.process_single_source(device_id, f"webcam_{device_id}")


def process_rtsp_stream(
    rtsp_url: str, 
    target_fps: float = 30.0,
    yolo_model: str = "yolov11n.pt",
    confidence: float = 0.5
) -> Iterator[Tuple[np.ndarray, Dict[str, Any], List[Detection]]]:
    """
    Convenience function for processing RTSP stream with YOLO detection.
    
    Args:
        rtsp_url: RTSP stream URL
        target_fps: Target processing FPS
        yolo_model: YOLO model to use
        confidence: Confidence threshold
    
    Yields:
        Tuple[np.ndarray, Dict[str, Any], List[Detection]]: 
        (frame, metadata, detections)
    """
    processor = FrameProcessor(
        target_fps=target_fps,
        yolo_model_path=yolo_model,
        confidence_threshold=confidence
    )
    yield from processor.process_single_source(rtsp_url, "rtsp_stream")


def process_video_file(
    file_path: str, 
    target_fps: float = 30.0,
    yolo_model: str = "yolov11n.pt",
    confidence: float = 0.5
) -> Iterator[Tuple[np.ndarray, Dict[str, Any], List[Detection]]]:
    """
    Convenience function for processing video file with YOLO detection.
    
    Args:
        file_path: Path to video file
        target_fps: Target processing FPS
        yolo_model: YOLO model to use
        confidence: Confidence threshold
    
    Yields:
        Tuple[np.ndarray, Dict[str, Any], List[Detection]]: 
        (frame, metadata, detections)
    """
    processor = FrameProcessor(
        target_fps=target_fps,
        yolo_model_path=yolo_model,
        confidence_threshold=confidence
    )
    yield from processor.process_single_source(file_path, f"file_{file_path}")


# Example usage and testing functions
def demo_real_time_processing():
    """
    Demonstration of real-time frame processing with YOLOv11 detection.
    
    This shows how frames are continuously read and processed for license plates.
    """
    print("Starting real-time ANPR processing demo...")
    print("Press 'q' to quit, 's' to save detected plates")
    
    processor = FrameProcessor(
        target_fps=30.0, 
        frame_skip=1,
        yolo_model_path="yolov11n.pt",
        confidence_threshold=0.5
    )
    
    try:
        for frame, metadata, detections in processor.process_single_source(0, "webcam_demo"):
            # Visualize detections on frame
            viz_frame = processor.visualize_detections(frame, detections)
            
            # Add processing stats overlay
            stats = processor.get_processing_stats()
            info_text = f"Frame: {metadata['frame_id']} | FPS: {stats['actual_fps']:.1f} | Plates: {len(detections)}"
            
            cv2.putText(
                viz_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Display frame
            cv2.imshow('ANPR Real-time Processing', viz_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and detections:
                # Save detected plates
                saved_count = 0
                for i, detection in enumerate(detections):
                    if detection.cropped_plate is not None:
                        filename = f"detected_plate_{metadata['frame_id']}_{i}.jpg"
                        cv2.imwrite(filename, detection.cropped_plate)
                        saved_count += 1
                print(f"Saved {saved_count} cropped plates")
            
            # Print detection info
            if detections:
                for i, det in enumerate(detections):
                    print(f"  Detection {i+1}: {det.license_plate} (conf: {det.confidence:.3f})")
    
    except KeyboardInterrupt:
        print("Processing interrupted by user")
    
    finally:
        cv2.destroyAllWindows()
        print("Demo finished")
        
        # Print final statistics
        final_stats = processor.get_processing_stats()
        print(f"Final processing stats: {final_stats}")


def demo_multi_source_processing():
    """
    Demonstration of multi-source processing with YOLO detection.
    """
    print("Starting multi-source ANPR processing demo...")
    
    # Define multiple sources
    sources = [
        (0, "webcam_0"),  # Primary webcam
        # ("rtsp://example.com/stream1", "cctv_1"),  # RTSP stream
        # ("sample_video.mp4", "video_file")  # Video file
    ]
    
    processor = FrameProcessor(
        target_fps=15.0, 
        frame_skip=2,
        yolo_model_path="yolov11n.pt",
        confidence_threshold=0.5
    )  # Lower FPS for multiple sources
    
    try:
        frame_counts = {}
        detection_counts = {}
        
        for frame, metadata, detections in processor.process_multiple_sources(sources):
            source_id = metadata['source_id']
            frame_counts[source_id] = frame_counts.get(source_id, 0) + 1
            detection_counts[source_id] = detection_counts.get(source_id, 0) + len(detections)
            
            print(f"Processed frame from {source_id}: {frame_counts[source_id]} (plates: {len(detections)})")
            
            # Print detection details
            for detection in detections:
                print(f"  -> {detection.license_plate} (conf: {detection.confidence:.3f})")
            
            # Break after some frames for demo
            total_frames = sum(frame_counts.values())
            if total_frames > 50:  # Process fewer frames for demo
                break
    
    except KeyboardInterrupt:
        print("Processing interrupted by user")
    
    finally:
        print(f"Final frame counts: {frame_counts}")
        print(f"Final detection counts: {detection_counts}")
        stats = processor.get_processing_stats()
        print(f"Processing stats: {stats}")


def demo_batch_processing(video_path: str):
    """
    Demonstration of batch video processing with YOLO detection.
    
    Args:
        video_path: Path to video file for processing
    """
    print(f"Starting batch processing of video: {video_path}")
    
    processor = FrameProcessor(
        target_fps=0,  # Process as fast as possible
        frame_skip=1,
        yolo_model_path="yolov11n.pt",
        confidence_threshold=0.5
    )
    
    all_detections = []
    frame_count = 0
    
    try:
        for frame, metadata, detections in processor.process_single_source(video_path, "batch_video"):
            frame_count += 1
            all_detections.extend(detections)
            
            # Print progress every 30 frames
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames, found {len(all_detections)} total plates")
            
            # Save cropped plates
            for i, detection in enumerate(detections):
                if detection.cropped_plate is not None:
                    filename = f"batch_plate_frame{frame_count:06d}_{i}.jpg"
                    cv2.imwrite(filename, detection.cropped_plate)
    
    except KeyboardInterrupt:
        print("Batch processing interrupted by user")
    
    finally:
        print(f"Batch processing complete:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Total plates detected: {len(all_detections)}")
        print(f"  Average plates per frame: {len(all_detections)/frame_count:.2f}")
        
        # Print final statistics
        final_stats = processor.get_processing_stats()
        print(f"Final processing stats: {final_stats}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("ANPR Processing System - Demo Options:")
        print("  python processor.py demo           # Real-time webcam demo")
        print("  python processor.py multi          # Multi-source demo")
        print("  python processor.py batch <path>   # Batch video processing")
        print("  python processor.py webcam [id]    # Process webcam")
        print("  python processor.py rtsp <url>     # Process RTSP stream")
        print("  python processor.py file <path>    # Process video file")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    try:
        if command == "demo":
            demo_real_time_processing()
        
        elif command == "multi":
            demo_multi_source_processing()
        
        elif command == "batch":
            if len(sys.argv) < 3:
                print("Please provide video file path")
                sys.exit(1)
            video_path = sys.argv[2]
            demo_batch_processing(video_path)
        
        elif command == "webcam":
            device_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            
            print(f"Processing webcam {device_id} with YOLOv11...")
            frame_count = 0
            for frame, metadata, detections in process_webcam_stream(device_id):
                frame_count += 1
                print(f"Frame {frame_count}: {len(detections)} plates detected")
                for det in detections:
                    print(f"  -> {det.license_plate} (conf: {det.confidence:.3f})")
                
                if frame_count >= 100:  # Process 100 frames for demo
                    break
        
        elif command == "rtsp":
            if len(sys.argv) < 3:
                print("Please provide RTSP URL")
                sys.exit(1)
            
            rtsp_url = sys.argv[2]
            
            print(f"Processing RTSP stream: {rtsp_url}")
            frame_count = 0
            for frame, metadata, detections in process_rtsp_stream(rtsp_url):
                frame_count += 1
                print(f"Frame {frame_count}: {len(detections)} plates detected")
                for det in detections:
                    print(f"  -> {det.license_plate} (conf: {det.confidence:.3f})")
                
                if frame_count >= 100:
                    break
        
        elif command == "file":
            if len(sys.argv) < 3:
                print("Please provide file path")
                sys.exit(1)
            
            file_path = sys.argv[2]
            
            print(f"Processing video file: {file_path}")
            frame_count = 0
            detections_list = list(process_video_file(file_path))
            
            for frame, metadata, detections in detections_list:
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames")
            
            total_detections = sum(len(dets) for _, _, dets in detections_list)
            print(f"Total frames: {len(detections_list)}, Total plates: {total_detections}")
        
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
