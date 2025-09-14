"""
Real-time update system for interactive dashboards.

This module provides WebSocket-based real-time updates, event streaming,
and live data synchronization for dashboard components.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from abc import ABC, abstractmethod
import pandas as pd

from .dashboard_framework import (
    DashboardEvent, DashboardEventHandler, DashboardEventType,
    InteractiveDashboard, DashboardController
)
from ..core.models import Dashboard, Chart


logger = logging.getLogger(__name__)


@dataclass
class ClientConnection:
    """Represents a connected client for real-time updates."""
    client_id: str
    websocket: Any  # WebSocket connection object
    subscribed_events: Set[str] = field(default_factory=set)
    last_ping: datetime = field(default_factory=datetime.now)
    dashboard_id: Optional[str] = None


@dataclass
class UpdateMessage:
    """Message format for real-time updates."""
    message_type: str
    dashboard_id: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    client_id: Optional[str] = None


class RealtimeEventBroadcaster(DashboardEventHandler):
    """
    Broadcasts dashboard events to connected clients in real-time.
    
    Handles WebSocket connections and manages event distribution
    to subscribed clients.
    """
    
    def __init__(self):
        self.clients: Dict[str, ClientConnection] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        self.broadcast_task: Optional[asyncio.Task] = None
    
    async def start_broadcaster(self) -> None:
        """Start the real-time event broadcaster."""
        if self.is_running:
            return
        
        self.is_running = True
        self.broadcast_task = asyncio.create_task(self._broadcast_loop())
        logger.info("Real-time event broadcaster started")
    
    async def stop_broadcaster(self) -> None:
        """Stop the real-time event broadcaster."""
        self.is_running = False
        
        if self.broadcast_task:
            self.broadcast_task.cancel()
            try:
                await self.broadcast_task
            except asyncio.CancelledError:
                pass
        
        # Close all client connections
        for client in list(self.clients.values()):
            await self.disconnect_client(client.client_id)
        
        logger.info("Real-time event broadcaster stopped")
    
    async def connect_client(self, client_id: str, websocket: Any, dashboard_id: Optional[str] = None) -> None:
        """Connect a new client for real-time updates."""
        client = ClientConnection(
            client_id=client_id,
            websocket=websocket,
            dashboard_id=dashboard_id
        )
        
        self.clients[client_id] = client
        
        # Send connection confirmation
        await self._send_to_client(client_id, UpdateMessage(
            message_type="connection_established",
            dashboard_id=dashboard_id or "",
            data={"client_id": client_id, "status": "connected"}
        ))
        
        logger.info(f"Client {client_id} connected for dashboard {dashboard_id}")
    
    async def disconnect_client(self, client_id: str) -> None:
        """Disconnect a client."""
        if client_id in self.clients:
            client = self.clients[client_id]
            
            try:
                if hasattr(client.websocket, 'close'):
                    await client.websocket.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket for client {client_id}: {e}")
            
            del self.clients[client_id]
            logger.info(f"Client {client_id} disconnected")
    
    def subscribe_client(self, client_id: str, event_types: List[str]) -> None:
        """Subscribe client to specific event types."""
        if client_id in self.clients:
            self.clients[client_id].subscribed_events.update(event_types)
            logger.info(f"Client {client_id} subscribed to events: {event_types}")
    
    def unsubscribe_client(self, client_id: str, event_types: List[str]) -> None:
        """Unsubscribe client from specific event types."""
        if client_id in self.clients:
            self.clients[client_id].subscribed_events.difference_update(event_types)
            logger.info(f"Client {client_id} unsubscribed from events: {event_types}")
    
    def handle_event(self, event: DashboardEvent) -> None:
        """Handle dashboard event and queue for broadcast."""
        if self.is_running:
            try:
                # Convert event to update message
                update_message = UpdateMessage(
                    message_type=event.event_type,
                    dashboard_id=event.data.get("dashboard_id", ""),
                    data=event.data
                )
                
                # Queue for broadcast (non-blocking)
                asyncio.create_task(self._queue_message(update_message))
            except Exception as e:
                logger.error(f"Error handling event for broadcast: {e}")
    
    async def _queue_message(self, message: UpdateMessage) -> None:
        """Queue message for broadcast."""
        try:
            await self.event_queue.put(message)
        except Exception as e:
            logger.error(f"Error queuing message: {e}")
    
    async def _broadcast_loop(self) -> None:
        """Main broadcast loop for processing queued messages."""
        while self.is_running:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._broadcast_message(message)
            except asyncio.TimeoutError:
                # Periodic cleanup and health check
                await self._cleanup_stale_connections()
            except Exception as e:
                logger.error(f"Error in broadcast loop: {e}")
    
    async def _broadcast_message(self, message: UpdateMessage) -> None:
        """Broadcast message to relevant clients."""
        disconnected_clients = []
        
        for client_id, client in self.clients.items():
            try:
                # Check if client is subscribed to this event type
                if (not client.subscribed_events or 
                    message.message_type in client.subscribed_events):
                    
                    # Check if message is for this client's dashboard
                    if (not message.dashboard_id or 
                        not client.dashboard_id or 
                        client.dashboard_id == message.dashboard_id):
                        
                        await self._send_to_client(client_id, message)
            
            except Exception as e:
                logger.warning(f"Error sending to client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect_client(client_id)
    
    async def _send_to_client(self, client_id: str, message: UpdateMessage) -> None:
        """Send message to specific client."""
        if client_id not in self.clients:
            return
        
        client = self.clients[client_id]
        
        try:
            message_json = json.dumps(asdict(message), default=str)
            
            if hasattr(client.websocket, 'send'):
                await client.websocket.send(message_json)
            elif hasattr(client.websocket, 'send_text'):
                await client.websocket.send_text(message_json)
            
            client.last_ping = datetime.now()
            
        except Exception as e:
            logger.warning(f"Failed to send message to client {client_id}: {e}")
            raise
    
    async def _cleanup_stale_connections(self) -> None:
        """Clean up stale client connections."""
        current_time = datetime.now()
        stale_clients = []
        
        for client_id, client in self.clients.items():
            # Consider connection stale if no activity for 5 minutes
            if (current_time - client.last_ping).total_seconds() > 300:
                stale_clients.append(client_id)
        
        for client_id in stale_clients:
            await self.disconnect_client(client_id)
    
    def get_connected_clients(self) -> List[str]:
        """Get list of connected client IDs."""
        return list(self.clients.keys())
    
    def get_client_count(self) -> int:
        """Get number of connected clients."""
        return len(self.clients)


class DataStreamManager:
    """
    Manages real-time data streams for dashboard updates.
    
    Handles data source monitoring, change detection, and
    automatic dashboard refresh triggers.
    """
    
    def __init__(self, dashboard_controller: DashboardController):
        self.dashboard_controller = dashboard_controller
        self.data_sources: Dict[str, Callable[[], pd.DataFrame]] = {}
        self.stream_intervals: Dict[str, int] = {}  # seconds
        self.stream_tasks: Dict[str, asyncio.Task] = {}
        self.last_data_hashes: Dict[str, str] = {}
        self.change_callbacks: List[Callable[[str, pd.DataFrame], None]] = []
        self.is_running = False
    
    def add_data_source(self, source_id: str, data_callback: Callable[[], pd.DataFrame], 
                       interval_seconds: int = 30) -> None:
        """Add a data source for monitoring."""
        self.data_sources[source_id] = data_callback
        self.stream_intervals[source_id] = interval_seconds
        
        logger.info(f"Added data source {source_id} with {interval_seconds}s interval")
    
    def remove_data_source(self, source_id: str) -> None:
        """Remove a data source."""
        if source_id in self.data_sources:
            # Stop streaming task if running
            if source_id in self.stream_tasks:
                self.stream_tasks[source_id].cancel()
                del self.stream_tasks[source_id]
            
            del self.data_sources[source_id]
            del self.stream_intervals[source_id]
            
            if source_id in self.last_data_hashes:
                del self.last_data_hashes[source_id]
            
            logger.info(f"Removed data source {source_id}")
    
    async def start_streaming(self) -> None:
        """Start monitoring all data sources."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start streaming task for each data source
        for source_id in self.data_sources:
            task = asyncio.create_task(self._stream_data_source(source_id))
            self.stream_tasks[source_id] = task
        
        logger.info(f"Started streaming for {len(self.data_sources)} data sources")
    
    async def stop_streaming(self) -> None:
        """Stop monitoring all data sources."""
        self.is_running = False
        
        # Cancel all streaming tasks
        for task in self.stream_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.stream_tasks:
            await asyncio.gather(*self.stream_tasks.values(), return_exceptions=True)
        
        self.stream_tasks.clear()
        logger.info("Stopped all data streaming")
    
    async def _stream_data_source(self, source_id: str) -> None:
        """Stream data from a specific source."""
        while self.is_running and source_id in self.data_sources:
            try:
                # Get fresh data
                data_callback = self.data_sources[source_id]
                new_data = data_callback()
                
                # Check if data has changed
                if self._has_data_changed(source_id, new_data):
                    logger.info(f"Data change detected in source {source_id}")
                    
                    # Update dashboard
                    updated_dashboard = self.dashboard_controller.refresh_dashboard(new_data)
                    
                    # Notify change callbacks
                    for callback in self.change_callbacks:
                        try:
                            callback(source_id, new_data)
                        except Exception as e:
                            logger.error(f"Error in change callback: {e}")
                
                # Wait for next interval
                interval = self.stream_intervals.get(source_id, 30)
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error streaming data source {source_id}: {e}")
                await asyncio.sleep(5)  # Wait before retry
    
    def _has_data_changed(self, source_id: str, new_data: pd.DataFrame) -> bool:
        """Check if data has changed since last check."""
        try:
            # Create hash of data for comparison
            data_hash = str(hash(pd.util.hash_pandas_object(new_data).sum()))
            
            if source_id not in self.last_data_hashes:
                self.last_data_hashes[source_id] = data_hash
                return True  # First time, consider as change
            
            if self.last_data_hashes[source_id] != data_hash:
                self.last_data_hashes[source_id] = data_hash
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking data change for {source_id}: {e}")
            return True  # Assume change on error
    
    def add_change_callback(self, callback: Callable[[str, pd.DataFrame], None]) -> None:
        """Add callback for data change notifications."""
        self.change_callbacks.append(callback)
    
    def get_active_sources(self) -> List[str]:
        """Get list of active data source IDs."""
        return list(self.data_sources.keys())


class RealtimeDashboardManager:
    """
    High-level manager for real-time dashboard functionality.
    
    Coordinates between dashboard framework, event broadcasting,
    and data streaming for complete real-time experience.
    """
    
    def __init__(self, dashboard_controller: DashboardController):
        self.dashboard_controller = dashboard_controller
        self.event_broadcaster = RealtimeEventBroadcaster()
        self.data_stream_manager = DataStreamManager(dashboard_controller)
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize real-time dashboard system."""
        if self.is_initialized:
            return
        
        # Setup event handling
        self.dashboard_controller.dashboard.add_event_handler(self.event_broadcaster)
        
        # Setup data change callback
        self.data_stream_manager.add_change_callback(self._on_data_change)
        
        # Start services
        await self.event_broadcaster.start_broadcaster()
        await self.data_stream_manager.start_streaming()
        
        self.is_initialized = True
        logger.info("Real-time dashboard manager initialized")
    
    async def shutdown(self) -> None:
        """Shutdown real-time dashboard system."""
        if not self.is_initialized:
            return
        
        await self.data_stream_manager.stop_streaming()
        await self.event_broadcaster.stop_broadcaster()
        
        self.is_initialized = False
        logger.info("Real-time dashboard manager shutdown")
    
    async def connect_client(self, client_id: str, websocket: Any, dashboard_id: Optional[str] = None) -> None:
        """Connect client for real-time updates."""
        await self.event_broadcaster.connect_client(client_id, websocket, dashboard_id)
        
        # Subscribe to all event types by default
        self.event_broadcaster.subscribe_client(client_id, [
            DashboardEventType.FILTER_CHANGED,
            DashboardEventType.DATA_UPDATED,
            DashboardEventType.CHART_SELECTED,
            DashboardEventType.REFRESH_REQUESTED
        ])
    
    async def disconnect_client(self, client_id: str) -> None:
        """Disconnect client."""
        await self.event_broadcaster.disconnect_client(client_id)
    
    def add_data_source(self, source_id: str, data_callback: Callable[[], pd.DataFrame], 
                       interval_seconds: int = 30) -> None:
        """Add real-time data source."""
        self.data_stream_manager.add_data_source(source_id, data_callback, interval_seconds)
    
    def remove_data_source(self, source_id: str) -> None:
        """Remove real-time data source."""
        self.data_stream_manager.remove_data_source(source_id)
    
    def _on_data_change(self, source_id: str, new_data: pd.DataFrame) -> None:
        """Handle data change from stream manager."""
        # This will trigger dashboard events that get broadcast to clients
        logger.info(f"Processing data change from source {source_id}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get real-time system status."""
        return {
            "initialized": self.is_initialized,
            "connected_clients": self.event_broadcaster.get_client_count(),
            "active_data_sources": len(self.data_stream_manager.get_active_sources()),
            "data_sources": self.data_stream_manager.get_active_sources()
        }


# WebSocket message handlers for dashboard interactions
class DashboardWebSocketHandler:
    """
    WebSocket message handler for dashboard interactions.
    
    Processes incoming WebSocket messages and coordinates
    dashboard operations and real-time updates.
    """
    
    def __init__(self, realtime_manager: RealtimeDashboardManager):
        self.realtime_manager = realtime_manager
        self.message_handlers = {
            "filter_update": self._handle_filter_update,
            "refresh_request": self._handle_refresh_request,
            "subscribe_events": self._handle_subscribe_events,
            "unsubscribe_events": self._handle_unsubscribe_events,
            "ping": self._handle_ping
        }
    
    async def handle_message(self, client_id: str, message: Dict[str, Any]) -> None:
        """Handle incoming WebSocket message."""
        message_type = message.get("type")
        
        if message_type in self.message_handlers:
            try:
                await self.message_handlers[message_type](client_id, message)
            except Exception as e:
                logger.error(f"Error handling message type {message_type}: {e}")
                await self._send_error(client_id, f"Error processing {message_type}: {str(e)}")
        else:
            await self._send_error(client_id, f"Unknown message type: {message_type}")
    
    async def _handle_filter_update(self, client_id: str, message: Dict[str, Any]) -> None:
        """Handle filter update request."""
        column = message.get("column")
        values = message.get("values", [])
        
        if not column:
            await self._send_error(client_id, "Missing column in filter update")
            return
        
        # Update filter through dashboard controller
        updated_dashboard = self.realtime_manager.dashboard_controller.apply_filter(column, values)
        
        # Send confirmation
        await self._send_response(client_id, {
            "type": "filter_updated",
            "column": column,
            "values": values,
            "dashboard_id": updated_dashboard.id
        })
    
    async def _handle_refresh_request(self, client_id: str, message: Dict[str, Any]) -> None:
        """Handle dashboard refresh request."""
        updated_dashboard = self.realtime_manager.dashboard_controller.refresh_dashboard()
        
        await self._send_response(client_id, {
            "type": "dashboard_refreshed",
            "dashboard_id": updated_dashboard.id,
            "timestamp": datetime.now().isoformat()
        })
    
    async def _handle_subscribe_events(self, client_id: str, message: Dict[str, Any]) -> None:
        """Handle event subscription request."""
        event_types = message.get("event_types", [])
        
        self.realtime_manager.event_broadcaster.subscribe_client(client_id, event_types)
        
        await self._send_response(client_id, {
            "type": "subscribed",
            "event_types": event_types
        })
    
    async def _handle_unsubscribe_events(self, client_id: str, message: Dict[str, Any]) -> None:
        """Handle event unsubscription request."""
        event_types = message.get("event_types", [])
        
        self.realtime_manager.event_broadcaster.unsubscribe_client(client_id, event_types)
        
        await self._send_response(client_id, {
            "type": "unsubscribed",
            "event_types": event_types
        })
    
    async def _handle_ping(self, client_id: str, message: Dict[str, Any]) -> None:
        """Handle ping message."""
        await self._send_response(client_id, {
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        })
    
    async def _send_response(self, client_id: str, data: Dict[str, Any]) -> None:
        """Send response message to client."""
        response_message = UpdateMessage(
            message_type=data.get("type", "response"),
            dashboard_id=data.get("dashboard_id", ""),
            data=data,
            client_id=client_id
        )
        
        await self.realtime_manager.event_broadcaster._send_to_client(client_id, response_message)
    
    async def _send_error(self, client_id: str, error_message: str) -> None:
        """Send error message to client."""
        await self._send_response(client_id, {
            "type": "error",
            "message": error_message,
            "timestamp": datetime.now().isoformat()
        })