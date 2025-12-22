"""
FastAPI Application Entry Point

Main application factory for the ANPR system backend.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging

from .api.routes import anpr, feeds, auth, system
from .core.config import get_settings
from .core.websocket import ConnectionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get application settings
settings = get_settings()

def create_application() -> FastAPI:
    """Application factory."""
    
    # Create FastAPI instance
    app = FastAPI(
        title="ANPR System API",
        description="Real-time Automatic Number Plate Recognition System",
        version="1.0.0",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routers
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
    app.include_router(anpr.router, prefix="/api/v1/anpr", tags=["anpr"])
    app.include_router(feeds.router, prefix="/api/v1/feeds", tags=["feeds"])
    app.include_router(system.router, prefix="/api/v1/system", tags=["system"])
    
    # Mount static files
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    # WebSocket manager
    connection_manager = ConnectionManager()
    
    @app.websocket("/ws/{client_id}")
    async def websocket_endpoint(websocket: WebSocket, client_id: str):
        """WebSocket endpoint for real-time communication."""
        await connection_manager.connect(websocket, client_id)
        try:
            while True:
                data = await websocket.receive_text()
                await connection_manager.send_personal_message(f"Message: {data}", client_id)
        except WebSocketDisconnect:
            connection_manager.disconnect(client_id)
    
    @app.on_event("startup")
    async def startup_event():
        """Application startup event."""
        logger.info("Starting ANPR System API...")
        logger.info(f"Debug mode: {settings.DEBUG}")
        logger.info(f"Database URL: {settings.DATABASE_URL}")
        
        # Initialize database connection
        # await database.connect()
        
        # Start background tasks
        # await start_background_tasks()
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Application shutdown event."""
        logger.info("Shutting down ANPR System API...")
        
        # Close database connection
        # await database.disconnect()
        
        # Stop background tasks
        # await stop_background_tasks()
    
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint."""
        return {
            "message": "ANPR System API",
            "version": "1.0.0",
            "status": "running"
        }
    
    @app.get("/health", tags=["system"])
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": "2025-01-01T00:00:00Z",
            "version": "1.0.0"
        }
    
    return app

# Create application instance
app = create_application()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )
