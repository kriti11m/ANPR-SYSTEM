"""
WebSocket Connection Manager

Manages WebSocket connections for real-time communication
between the backend and frontend clients.
"""

from typing import Dict, List
from fastapi import WebSocket, WebSocketDisconnect
import json
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        # Active connections: {client_id: websocket}
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Client subscriptions: {client_id: [subscription_types]}
        self.subscriptions: Dict[str, List[str]] = {}
        
        # Connection metadata
        self.connection_metadata: Dict[str, dict] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = []
        self.connection_metadata[client_id] = {
            "connected_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "message_count": 0
        }
        
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
        
        # Send connection confirmation
        await self.send_personal_message({
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat()
        }, client_id)
    
    def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        
        if client_id in self.connection_metadata:
            del self.connection_metadata[client_id]
        
        logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, client_id: str):
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                
                # Add timestamp to message
                message["timestamp"] = datetime.utcnow().isoformat()
                
                await websocket.send_text(json.dumps(message))
                
                # Update metadata
                if client_id in self.connection_metadata:
                    self.connection_metadata[client_id]["last_activity"] = datetime.utcnow().isoformat()
                    self.connection_metadata[client_id]["message_count"] += 1
                
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast_message(self, message: dict, subscription_type: str = None):
        """Broadcast a message to all connected clients or subscribers."""
        if not self.active_connections:
            return
        
        # Add timestamp to message
        message["timestamp"] = datetime.utcnow().isoformat()
        message_text = json.dumps(message)
        
        # Determine target clients
        target_clients = []
        
        if subscription_type:
            # Send only to clients subscribed to this type
            for client_id, subscriptions in self.subscriptions.items():
                if subscription_type in subscriptions and client_id in self.active_connections:
                    target_clients.append(client_id)
        else:
            # Send to all connected clients
            target_clients = list(self.active_connections.keys())
        
        # Send messages
        failed_clients = []
        
        for client_id in target_clients:
            try:
                websocket = self.active_connections[client_id]
                await websocket.send_text(message_text)
                
                # Update metadata
                if client_id in self.connection_metadata:
                    self.connection_metadata[client_id]["last_activity"] = datetime.utcnow().isoformat()
                    self.connection_metadata[client_id]["message_count"] += 1
                    
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                failed_clients.append(client_id)
        
        # Clean up failed connections
        for client_id in failed_clients:
            self.disconnect(client_id)
        
        logger.debug(f"Broadcast message to {len(target_clients)} clients")
    
    async def subscribe_client(self, client_id: str, subscription_type: str):
        """Subscribe a client to specific message types."""
        if client_id in self.subscriptions:
            if subscription_type not in self.subscriptions[client_id]:
                self.subscriptions[client_id].append(subscription_type)
                
                await self.send_personal_message({
                    "type": "subscription_confirmed",
                    "subscription_type": subscription_type
                }, client_id)
                
                logger.info(f"Client {client_id} subscribed to {subscription_type}")
    
    async def unsubscribe_client(self, client_id: str, subscription_type: str):
        """Unsubscribe a client from specific message types."""
        if client_id in self.subscriptions:
            if subscription_type in self.subscriptions[client_id]:
                self.subscriptions[client_id].remove(subscription_type)
                
                await self.send_personal_message({
                    "type": "subscription_cancelled",
                    "subscription_type": subscription_type
                }, client_id)
                
                logger.info(f"Client {client_id} unsubscribed from {subscription_type}")
    
    async def send_detection_result(self, detection_data: dict):
        """Send detection result to subscribed clients."""
        message = {
            "type": "detection_result",
            "data": detection_data
        }
        await self.broadcast_message(message, "detections")
    
    async def send_system_status(self, status_data: dict):
        """Send system status to subscribed clients."""
        message = {
            "type": "system_status",
            "data": status_data
        }
        await self.broadcast_message(message, "system_status")
    
    async def send_feed_update(self, feed_data: dict):
        """Send feed status update to subscribed clients."""
        message = {
            "type": "feed_update",
            "data": feed_data
        }
        await self.broadcast_message(message, "feeds")
    
    def get_connection_stats(self) -> dict:
        """Get connection statistics."""
        return {
            "total_connections": len(self.active_connections),
            "active_clients": list(self.active_connections.keys()),
            "subscriptions_summary": {
                client_id: len(subs) for client_id, subs in self.subscriptions.items()
            },
            "connection_metadata": self.connection_metadata
        }
    
    async def handle_client_message(self, websocket: WebSocket, client_id: str, message: str):
        """Handle incoming messages from clients."""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "subscribe":
                subscription_type = data.get("subscription_type")
                if subscription_type:
                    await self.subscribe_client(client_id, subscription_type)
            
            elif message_type == "unsubscribe":
                subscription_type = data.get("subscription_type")
                if subscription_type:
                    await self.unsubscribe_client(client_id, subscription_type)
            
            elif message_type == "ping":
                await self.send_personal_message({"type": "pong"}, client_id)
            
            elif message_type == "get_stats":
                stats = self.get_connection_stats()
                await self.send_personal_message({
                    "type": "connection_stats",
                    "data": stats
                }, client_id)
            
            else:
                logger.warning(f"Unknown message type from {client_id}: {message_type}")
                await self.send_personal_message({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                }, client_id)
        
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from {client_id}: {message}")
            await self.send_personal_message({
                "type": "error",
                "message": "Invalid JSON format"
            }, client_id)
        
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")
            await self.send_personal_message({
                "type": "error",
                "message": "Server error processing message"
            }, client_id)


# Global connection manager instance
connection_manager = ConnectionManager()
