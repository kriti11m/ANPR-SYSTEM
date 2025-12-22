"""
Video Capture Module for ANPR System

This module provides a unified interface for capturing video frames from multiple sources:
- Webcam (local camera)
- RTSP streams (IP cameras, CCTV)
- Video files (MP4, AVI, etc.)

The module continuously yields frames for real-time processing with YOLO.
"""

import cv2
import logging
import time
import threading
from typing import Optional, Union, Iterator, Tuple
from queue import Queue, Empty
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoCapture:
    """
    Unified video capture class supporting multiple input sources.
    
    Supports:
    - Webcam capture (source as int)
    - RTSP streams (source as rtsp:// URL)
    - Video files (source as file path)
    
    Features:
    - Thread-safe frame reading
    - Automatic reconnection for network streams
    - Frame buffering to prevent blocking
    - FPS control and monitoring
    """
    
    def __init__(
        self,
        source: Union[int, str],
        buffer_size: int = 10,
        retry_delay: float = 5.0,
        timeout: float = 30.0
    ):
        """
        Initialize video capture.
        
        Args:
            source: Video source (int for webcam, str for file/stream)
            buffer_size: Maximum number of frames to buffer
            retry_delay: Delay between reconnection attempts (seconds)
            timeout: Timeout for stream connections (seconds)
        """
        self.source = source
        self.buffer_size = buffer_size
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        # Internal state
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_queue: Queue = Queue(maxsize=buffer_size)
        self.capture_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.last_frame_time = 0
        self.fps_actual = 0.0
        
        # Statistics
        self.frames_captured = 0
        self.frames_dropped = 0
        self.connection_errors = 0
        
        self._initialize_capture()
    
    def _initialize_capture(self) -> bool:
        """Initialize the video capture device/stream."""
        try:
            logger.info(f"Initializing capture for source: {self.source}")
            
            # Create VideoCapture object
            self.cap = cv2.VideoCapture(self.source)
            
            # Set timeout for network streams
            if isinstance(self.source, str) and self.source.startswith(('rtsp://', 'http://')):
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout * 1000)
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.timeout * 1000)
            
            # Check if capture is opened successfully
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False
            
            # Get video properties
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Video source opened successfully:")
            logger.info(f"  Resolution: {width}x{height}")
            logger.info(f"  FPS: {fps}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing capture: {e}")
            self.connection_errors += 1
            return False
    
    def start(self) -> bool:
        """Start the frame capture thread."""
        if self.is_running:
            logger.warning("Capture is already running")
            return True
        
        if not self.cap or not self.cap.isOpened():
            if not self._initialize_capture():
                return False
        
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        
        logger.info("Frame capture started")
        return True
    
    def stop(self):
        """Stop the frame capture thread."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        self._cleanup()
        logger.info("Frame capture stopped")
    
    def _capture_frames(self):
        """Main capture loop running in separate thread."""
        consecutive_failures = 0
        max_failures = 10
        
        while self.is_running:
            try:
                if not self.cap or not self.cap.isOpened():
                    if not self._reconnect():
                        time.sleep(self.retry_delay)
                        continue
                
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    logger.warning(f"Failed to read frame (attempt {consecutive_failures})")
                    
                    if consecutive_failures >= max_failures:
                        logger.error("Too many consecutive failures, attempting reconnection")
                        self._reconnect()
                        consecutive_failures = 0
                    
                    time.sleep(0.1)
                    continue
                
                # Reset failure counter on successful read
                consecutive_failures = 0
                
                # Update statistics
                current_time = time.time()
                if self.last_frame_time > 0:
                    time_diff = current_time - self.last_frame_time
                    self.fps_actual = 1.0 / time_diff if time_diff > 0 else 0
                
                self.last_frame_time = current_time
                self.frames_captured += 1
                
                # Add frame to queue (non-blocking)
                try:
                    self.frame_queue.put(frame, block=False)
                except:
                    # Queue is full, drop oldest frame
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame, block=False)
                        self.frames_dropped += 1
                    except Empty:
                        pass
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(1.0)
    
    def _reconnect(self) -> bool:
        """Attempt to reconnect to the video source."""
        logger.info("Attempting to reconnect...")
        
        self._cleanup()
        time.sleep(self.retry_delay)
        
        return self._initialize_capture()
    
    def _cleanup(self):
        """Clean up capture resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read the latest frame from the capture buffer.
        
        Returns:
            numpy.ndarray: Latest frame or None if no frame available
        """
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None
    
    def get_frame_stream(self) -> Iterator[np.ndarray]:
        """
        Generator that continuously yields frames.
        
        This is the main method for real-time frame processing.
        
        Yields:
            numpy.ndarray: Video frames for YOLO processing
        """
        if not self.start():
            logger.error("Failed to start video capture")
            return
        
        try:
            frame_timeout = 1.0 / 30.0  # 30 FPS timeout
            
            while self.is_running:
                try:
                    frame = self.frame_queue.get(timeout=frame_timeout)
                    if frame is not None:
                        yield frame
                except Empty:
                    # No frame available, continue waiting
                    continue
                except Exception as e:
                    logger.error(f"Error yielding frame: {e}")
                    break
        
        finally:
            self.stop()
    
    def get_properties(self) -> dict:
        """Get video source properties."""
        if not self.cap or not self.cap.isOpened():
            return {}
        
        return {
            'fps_nominal': self.cap.get(cv2.CAP_PROP_FPS),
            'fps_actual': self.fps_actual,
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frames_captured': self.frames_captured,
            'frames_dropped': self.frames_dropped,
            'connection_errors': self.connection_errors,
            'buffer_size': self.frame_queue.qsize(),
            'is_running': self.is_running
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class MultiStreamCapture:
    """
    Capture from multiple video sources simultaneously.
    
    Useful for processing multiple CCTV cameras or video files.
    """
    
    def __init__(self, sources: list, buffer_size: int = 10):
        """
        Initialize multi-stream capture.
        
        Args:
            sources: List of video sources
            buffer_size: Buffer size for each stream
        """
        self.sources = sources
        self.captures = {}
        self.buffer_size = buffer_size
        
        # Initialize captures
        for i, source in enumerate(sources):
            capture_id = f"stream_{i}"
            self.captures[capture_id] = VideoCapture(
                source=source,
                buffer_size=buffer_size
            )
    
    def start_all(self) -> bool:
        """Start all video captures."""
        success = True
        for capture_id, capture in self.captures.items():
            if not capture.start():
                logger.error(f"Failed to start capture: {capture_id}")
                success = False
        
        return success
    
    def stop_all(self):
        """Stop all video captures."""
        for capture in self.captures.values():
            capture.stop()
    
    def read_all_frames(self) -> dict:
        """Read latest frame from all streams."""
        frames = {}
        for capture_id, capture in self.captures.items():
            frame = capture.read_frame()
            if frame is not None:
                frames[capture_id] = frame
        
        return frames
    
    def get_stream_iterator(self, stream_id: str) -> Iterator[np.ndarray]:
        """Get frame iterator for specific stream."""
        if stream_id in self.captures:
            return self.captures[stream_id].get_frame_stream()
        else:
            raise ValueError(f"Stream not found: {stream_id}")
    
    def get_all_properties(self) -> dict:
        """Get properties for all streams."""
        properties = {}
        for capture_id, capture in self.captures.items():
            properties[capture_id] = capture.get_properties()
        
        return properties


# Usage examples and utility functions
def test_webcam(device_id: int = 0, duration: int = 10):
    """Test webcam capture for specified duration."""
    print(f"Testing webcam {device_id} for {duration} seconds...")
    
    with VideoCapture(device_id) as capture:
        start_time = time.time()
        frame_count = 0
        
        for frame in capture.get_frame_stream():
            frame_count += 1
            
            # Display frame (optional)
            cv2.imshow('Webcam Test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Check duration
            if time.time() - start_time >= duration:
                break
        
        cv2.destroyAllWindows()
        
        properties = capture.get_properties()
        print(f"Captured {frame_count} frames")
        print(f"Properties: {properties}")


def test_rtsp_stream(rtsp_url: str, duration: int = 10):
    """Test RTSP stream capture."""
    print(f"Testing RTSP stream: {rtsp_url}")
    
    with VideoCapture(rtsp_url) as capture:
        start_time = time.time()
        frame_count = 0
        
        for frame in capture.get_frame_stream():
            frame_count += 1
            
            # Process frame (placeholder)
            print(f"Frame {frame_count}: {frame.shape}")
            
            # Check duration
            if time.time() - start_time >= duration:
                break
        
        properties = capture.get_properties()
        print(f"Captured {frame_count} frames")
        print(f"Properties: {properties}")


def test_video_file(file_path: str):
    """Test video file processing."""
    print(f"Testing video file: {file_path}")
    
    with VideoCapture(file_path) as capture:
        frame_count = 0
        
        for frame in capture.get_frame_stream():
            frame_count += 1
            
            # Process frame (placeholder)
            if frame_count % 30 == 0:  # Print every 30 frames
                print(f"Processed frame {frame_count}: {frame.shape}")
        
        properties = capture.get_properties()
        print(f"Total frames processed: {frame_count}")
        print(f"Properties: {properties}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python video_capture.py webcam [device_id]")
        print("  python video_capture.py rtsp <rtsp_url>")
        print("  python video_capture.py file <file_path>")
        sys.exit(1)
    
    source_type = sys.argv[1].lower()
    
    if source_type == "webcam":
        device_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        test_webcam(device_id)
    
    elif source_type == "rtsp":
        if len(sys.argv) < 3:
            print("Please provide RTSP URL")
            sys.exit(1)
        test_rtsp_stream(sys.argv[2])
    
    elif source_type == "file":
        if len(sys.argv) < 3:
            print("Please provide file path")
            sys.exit(1)
        test_video_file(sys.argv[2])
    
    else:
        print(f"Unknown source type: {source_type}")
        sys.exit(1)
