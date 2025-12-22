"""
Video Feed Management API Routes

Handles video feed sources (webcams, RTSP streams, video files).
"""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
import uuid

from .auth import get_current_user

# Initialize router
router = APIRouter()


# Pydantic models
class FeedCreate(BaseModel):
    name: str
    source_type: str  # "webcam", "rtsp", "file"
    source_url: str
    description: Optional[str] = None
    is_active: bool = True


class FeedUpdate(BaseModel):
    name: Optional[str] = None
    source_url: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


class Feed(BaseModel):
    id: str
    name: str
    source_type: str
    source_url: str
    description: Optional[str] = None
    is_active: bool
    created_at: datetime
    last_activity: Optional[datetime] = None
    status: str = "inactive"  # "active", "inactive", "error"
    error_message: Optional[str] = None


class FeedStats(BaseModel):
    feed_id: str
    fps: float
    frame_count: int
    detection_count: int
    last_detection: Optional[datetime] = None
    uptime: float  # seconds


# Mock database for feeds
mock_feeds = {}


@router.get("/", response_model=List[Feed])
async def get_feeds(
    is_active: Optional[bool] = None,
    source_type: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get all video feeds with optional filtering."""
    feeds = list(mock_feeds.values())
    
    # Apply filters
    if is_active is not None:
        feeds = [f for f in feeds if f.is_active == is_active]
    
    if source_type:
        feeds = [f for f in feeds if f.source_type == source_type]
    
    # Sort by created date
    feeds.sort(key=lambda x: x.created_at, reverse=True)
    
    return feeds


@router.post("/", response_model=Feed)
async def create_feed(
    feed_data: FeedCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new video feed."""
    # Validate source type
    valid_source_types = ["webcam", "rtsp", "file"]
    if feed_data.source_type not in valid_source_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid source_type. Must be one of: {valid_source_types}"
        )
    
    # Check if feed with same source already exists
    for feed in mock_feeds.values():
        if feed.source_url == feed_data.source_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Feed with this source URL already exists"
            )
    
    # Create new feed
    feed_id = str(uuid.uuid4())
    new_feed = Feed(
        id=feed_id,
        name=feed_data.name,
        source_type=feed_data.source_type,
        source_url=feed_data.source_url,
        description=feed_data.description,
        is_active=feed_data.is_active,
        created_at=datetime.utcnow(),
        status="inactive"
    )
    
    mock_feeds[feed_id] = new_feed
    
    # TODO: Initialize video capture for this feed
    # if new_feed.is_active:
    #     await start_feed_processing(feed_id)
    
    return new_feed


@router.get("/{feed_id}", response_model=Feed)
async def get_feed(
    feed_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get specific feed by ID."""
    if feed_id not in mock_feeds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feed not found"
        )
    
    return mock_feeds[feed_id]


@router.put("/{feed_id}", response_model=Feed)
async def update_feed(
    feed_id: str,
    feed_update: FeedUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update feed configuration."""
    if feed_id not in mock_feeds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feed not found"
        )
    
    feed = mock_feeds[feed_id]
    
    # Update fields
    if feed_update.name is not None:
        feed.name = feed_update.name
    
    if feed_update.source_url is not None:
        # Check if new URL conflicts with existing feeds
        for other_feed in mock_feeds.values():
            if other_feed.id != feed_id and other_feed.source_url == feed_update.source_url:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Feed with this source URL already exists"
                )
        feed.source_url = feed_update.source_url
    
    if feed_update.description is not None:
        feed.description = feed_update.description
    
    if feed_update.is_active is not None:
        was_active = feed.is_active
        feed.is_active = feed_update.is_active
        
        # TODO: Start/stop feed processing based on status change
        # if not was_active and feed.is_active:
        #     await start_feed_processing(feed_id)
        # elif was_active and not feed.is_active:
        #     await stop_feed_processing(feed_id)
    
    return feed


@router.delete("/{feed_id}")
async def delete_feed(
    feed_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a video feed."""
    if feed_id not in mock_feeds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feed not found"
        )
    
    # TODO: Stop feed processing if active
    # await stop_feed_processing(feed_id)
    
    del mock_feeds[feed_id]
    
    return {"message": "Feed deleted successfully"}


@router.post("/{feed_id}/start")
async def start_feed(
    feed_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Start processing a video feed."""
    if feed_id not in mock_feeds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feed not found"
        )
    
    feed = mock_feeds[feed_id]
    
    if feed.status == "active":
        return {"message": "Feed is already active"}
    
    try:
        # TODO: Start video capture and processing
        # await start_feed_processing(feed_id)
        
        feed.status = "active"
        feed.is_active = True
        feed.last_activity = datetime.utcnow()
        feed.error_message = None
        
        return {"message": "Feed started successfully"}
    
    except Exception as e:
        feed.status = "error"
        feed.error_message = str(e)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start feed: {str(e)}"
        )


@router.post("/{feed_id}/stop")
async def stop_feed(
    feed_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Stop processing a video feed."""
    if feed_id not in mock_feeds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feed not found"
        )
    
    feed = mock_feeds[feed_id]
    
    if feed.status == "inactive":
        return {"message": "Feed is already inactive"}
    
    try:
        # TODO: Stop video capture and processing
        # await stop_feed_processing(feed_id)
        
        feed.status = "inactive"
        feed.is_active = False
        
        return {"message": "Feed stopped successfully"}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop feed: {str(e)}"
        )


@router.get("/{feed_id}/stats", response_model=FeedStats)
async def get_feed_stats(
    feed_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get statistics for a specific feed."""
    if feed_id not in mock_feeds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feed not found"
        )
    
    # TODO: Get real statistics from video processing
    # For now, return mock stats
    return FeedStats(
        feed_id=feed_id,
        fps=25.5,
        frame_count=12345,
        detection_count=89,
        last_detection=datetime.utcnow(),
        uptime=3600.0  # 1 hour
    )


@router.post("/{feed_id}/test")
async def test_feed_connection(
    feed_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Test connection to a video feed."""
    if feed_id not in mock_feeds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feed not found"
        )
    
    feed = mock_feeds[feed_id]
    
    try:
        # TODO: Test actual connection to feed source
        # success, error_msg = await test_feed_source(feed.source_url)
        
        # Mock test result
        success = True
        error_msg = None
        
        if success:
            return {
                "status": "success",
                "message": "Feed connection test successful",
                "response_time": 150  # ms
            }
        else:
            return {
                "status": "error",
                "message": f"Feed connection test failed: {error_msg}",
                "response_time": None
            }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Test failed: {str(e)}"
        )
