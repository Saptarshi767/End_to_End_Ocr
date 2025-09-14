"""
Tests for real-time dashboard updates and WebSocket functionality.

This module tests real-time event broadcasting, data streaming,
WebSocket message handling, and live dashboard synchronization.
"""

import pytest
import asyncio
import json
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from src.visualization.realtime_updates import (
    RealtimeEventBroadcaster, DataStreamManager, RealtimeDashboardManager,
    DashboardWebSocketHandler, UpdateMessage, ClientConnection
)
from src.visualization.dashboard_framework import (
    InteractiveDashboard, DashboardController, DashboardEvent, DashboardEventType
)
from src.core.models import Dashboard


@pytest.mark.asyncio
class TestRealtimeEventBroadcaster:
    """Comprehensive tests for RealtimeEventBroadcaster."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.broadcaster = RealtimeEventBroadcaster()
    
    async def test_broadcaster_lifecycle(self):
        """Test broadcaster start/stop lifecycle."""
        # Initially not running
        assert self.broadcaster.is_running is False
        assert self.broadcaster.broadcast_task is None
        
        # Start broadcaster
        await self.broadcaster.start_broadcaster()
        assert self.broadcaster.is_running is True
        assert self.broadcaster.broadcast_task is not None
        
        # Stop broadcaster
        await self.broadcaster.stop_broadcaster()
        assert self.broadcaster.is_running is False
    
    async def test_client_connection_lifecycle(self):
        """Test complete client connection lifecycle."""
        # Mock WebSocket with proper async methods
        mock_websocket = AsyncMock()
        mock_websocket.send = AsyncMock()
        mock_websocket.close = AsyncMock()
        
        await self.broadcaster.start_broadcaster()
        
        # Connect client
        await self.broadcaster.connect_client("client1", mock_websocket, "dashboard1")
        
        # Verify client is connected
        assert "client1" in self.broadcaster.clients
        client = self.broadcaster.clients["client1"]
        assert client.client_id == "client1"
        assert client.dashboard_id == "dashboard1"
        assert client.websocket == mock_websocket
        
        # Verify connection message was sent
        mock_websocket.send.assert_called_once()
        
        # Disconnect client
        await self.broadcaster.disconnect_client("client1")
        
        # Verify client is removed
        assert "client1" not in self.broadcaster.clients
        mock_websocket.close.assert_called_once()
        
        await self.broadcaster.stop_broadcaster()
    
    async def test_event_subscription_management(self):
        """Test event subscription and unsubscription."""
        mock_websocket = AsyncMock()
        await self.broadcaster.connect_client("client1", mock_websocket)
        
        # Subscribe to events
        event_types = ["filter_changed", "data_updated"]
        self.broadcaster.subscribe_client("client1", event_types)
        
        client = self.broadcaster.clients["client1"]
        assert "filter_changed" in client.subscribed_events
        assert "data_updated" in client.subscribed_events
        
        # Unsubscribe from some events
        self.broadcaster.unsubscribe_client("client1", ["filter_changed"])
        
        assert "filter_changed" not in client.subscribed_events
        assert "data_updated" in client.subscribed_events
        
        await self.broadcaster.disconnect_client("client1")
    
    async def test_message_broadcasting(self):
        """Test message broadcasting to subscribed clients."""
        # Setup multiple clients
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        mock_ws3 = AsyncMock()
        
        await self.broadcaster.start_broadcaster()
        
        await self.broadcaster.connect_client("client1", mock_ws1, "dashboard1")
        await self.broadcaster.connect_client("client2", mock_ws2, "dashboard1")
        await self.broadcaster.connect_client("client3", mock_ws3, "dashboard2")
        
        # Subscribe clients to different events
        self.broadcaster.subscribe_client("client1", ["filter_changed"])
        self.broadcaster.subscribe_client("client2", ["filter_changed", "data_updated"])
        self.broadcaster.subscribe_client("client3", ["data_updated"])
        
        # Create and broadcast message
        message = UpdateMessage(
            message_type="filter_changed",
            dashboard_id="dashboard1",
            data={"column": "test", "values": ["A", "B"]}
        )
        
        await self.broadcaster._broadcast_message(message)
        
        # Give time for async processing
        await asyncio.sleep(0.1)
        
        # Verify correct clients received message
        # client1 and client2 should receive (subscribed to filter_changed and same dashboard)
        assert mock_ws1.send.call_count >= 1  # Connection + broadcast message
        assert mock_ws2.send.call_count >= 1  # Connection + broadcast message
        
        # client3 should only receive connection message (different event type)
        # Reset call counts to test specific message
        mock_ws1.reset_mock()
        mock_ws2.reset_mock()
        mock_ws3.reset_mock()
        
        # Send another message
        await self.broadcaster._broadcast_message(message)
        await asyncio.sleep(0.1)
        
        # Now only client1 and client2 should receive the filter_changed message
        assert mock_ws1.send.call_count == 1
        assert mock_ws2.send.call_count == 1
        assert mock_ws3.send.call_count == 0  # Not subscribed to filter_changed
        
        await self.broadcaster.stop_broadcaster()
    
    async def test_dashboard_filtering(self):
        """Test message filtering by dashboard ID."""
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        
        await self.broadcaster.start_broadcaster()
        
        await self.broadcaster.connect_client("client1", mock_ws1, "dashboard1")
        await self.broadcaster.connect_client("client2", mock_ws2, "dashboard2")
        
        # Subscribe both to same event type
        self.broadcaster.subscribe_client("client1", ["data_updated"])
        self.broadcaster.subscribe_client("client2", ["data_updated"])
        
        # Send message for dashboard1 only
        message = UpdateMessage(
            message_type="data_updated",
            dashboard_id="dashboard1",
            data={"refresh_type": "manual"}
        )
        
        # Reset mocks to ignore connection messages
        mock_ws1.reset_mock()
        mock_ws2.reset_mock()
        
        await self.broadcaster._broadcast_message(message)
        await asyncio.sleep(0.1)
        
        # Only client1 should receive message
        assert mock_ws1.send.call_count == 1
        assert mock_ws2.send.call_count == 0
        
        await self.broadcaster.stop_broadcaster()
    
    async def test_stale_connection_cleanup(self):
        """Test cleanup of stale connections."""
        mock_websocket = AsyncMock()
        
        await self.broadcaster.connect_client("client1", mock_websocket)
        
        # Simulate stale connection by setting old last_ping
        client = self.broadcaster.clients["client1"]
        client.last_ping = datetime.now() - timedelta(minutes=10)
        
        # Trigger cleanup
        await self.broadcaster._cleanup_stale_connections()
        
        # Client should be removed
        assert "client1" not in self.broadcaster.clients
    
    async def test_failed_message_handling(self):
        """Test handling of failed message sends."""
        # Mock WebSocket that fails on send
        mock_websocket = AsyncMock()
        mock_websocket.send.side_effect = Exception("Connection lost")
        
        await self.broadcaster.start_broadcaster()
        await self.broadcaster.connect_client("client1", mock_websocket)
        
        # Try to send message
        message = UpdateMessage(
            message_type="test",
            dashboard_id="dashboard1",
            data={}
        )
        
        # Should handle exception gracefully and remove client
        await self.broadcaster._broadcast_message(message)
        await asyncio.sleep(0.1)
        
        # Client should be disconnected due to send failure
        assert "client1" not in self.broadcaster.clients
        
        await self.broadcaster.stop_broadcaster()


@pytest.mark.asyncio
class TestDataStreamManager:
    """Comprehensive tests for DataStreamManager."""
    
    def setup_method(self):
        """Setup test fixtures."""
        dashboard_framework = InteractiveDashboard()
        controller = DashboardController(dashboard_framework)
        self.stream_manager = DataStreamManager(controller)
        
        # Sample data for testing
        self.sample_data = pd.DataFrame({
            "category": ["A", "B", "C"],
            "value": [10, 20, 30]
        })
    
    def test_data_source_management(self):
        """Test adding and removing data sources."""
        def mock_callback():
            return self.sample_data
        
        # Add data source
        self.stream_manager.add_data_source("source1", mock_callback, 10)
        
        assert "source1" in self.stream_manager.data_sources
        assert self.stream_manager.stream_intervals["source1"] == 10
        assert len(self.stream_manager.get_active_sources()) == 1
        
        # Add another source
        self.stream_manager.add_data_source("source2", mock_callback, 20)
        assert len(self.stream_manager.get_active_sources()) == 2
        
        # Remove source
        self.stream_manager.remove_data_source("source1")
        
        assert "source1" not in self.stream_manager.data_sources
        assert len(self.stream_manager.get_active_sources()) == 1
    
    async def test_streaming_lifecycle(self):
        """Test streaming start/stop lifecycle."""
        def mock_callback():
            return self.sample_data
        
        self.stream_manager.add_data_source("source1", mock_callback, 1)
        
        # Start streaming
        await self.stream_manager.start_streaming()
        
        assert self.stream_manager.is_running is True
        assert len(self.stream_manager.stream_tasks) == 1
        assert "source1" in self.stream_manager.stream_tasks
        
        # Stop streaming
        await self.stream_manager.stop_streaming()
        
        assert self.stream_manager.is_running is False
        assert len(self.stream_manager.stream_tasks) == 0
    
    def test_data_change_detection(self):
        """Test data change detection algorithm."""
        # Test with identical data
        data1 = pd.DataFrame({"test": [1, 2, 3]})
        data2 = pd.DataFrame({"test": [1, 2, 3]})
        
        # First call should return True (no previous data)
        assert self.stream_manager._has_data_changed("source1", data1) is True
        
        # Same data should return False
        assert self.stream_manager._has_data_changed("source1", data2) is False
        
        # Different data should return True
        data3 = pd.DataFrame({"test": [4, 5, 6]})
        assert self.stream_manager._has_data_changed("source1", data3) is True
        
        # Test with different structure
        data4 = pd.DataFrame({"different": [1, 2, 3]})
        assert self.stream_manager._has_data_changed("source1", data4) is True
    
    async def test_data_streaming_with_changes(self):
        """Test data streaming with actual data changes."""
        call_count = 0
        
        def changing_callback():
            nonlocal call_count
            call_count += 1
            return pd.DataFrame({"value": [call_count]})
        
        # Mock change callback
        change_callback = Mock()
        self.stream_manager.add_change_callback(change_callback)
        
        self.stream_manager.add_data_source("source1", changing_callback, 0.1)  # Very short interval
        
        # Start streaming
        await self.stream_manager.start_streaming()
        
        # Wait for a few iterations
        await asyncio.sleep(0.5)
        
        # Stop streaming
        await self.stream_manager.stop_streaming()
        
        # Change callback should have been called multiple times
        assert change_callback.call_count > 0
    
    async def test_streaming_error_handling(self):
        """Test error handling in data streaming."""
        def failing_callback():
            raise Exception("Data source error")
        
        self.stream_manager.add_data_source("failing_source", failing_callback, 0.1)
        
        # Start streaming (should handle errors gracefully)
        await self.stream_manager.start_streaming()
        
        # Wait a bit to let it try and fail
        await asyncio.sleep(0.2)
        
        # Should still be running despite errors
        assert self.stream_manager.is_running is True
        
        await self.stream_manager.stop_streaming()
    
    def test_change_callback_management(self):
        """Test change callback management."""
        callback1 = Mock()
        callback2 = Mock()
        
        # Add callbacks
        self.stream_manager.add_change_callback(callback1)
        self.stream_manager.add_change_callback(callback2)
        
        assert len(self.stream_manager.change_callbacks) == 2
        
        # Simulate change notification
        for callback in self.stream_manager.change_callbacks:
            callback("source1", self.sample_data)
        
        callback1.assert_called_once_with("source1", self.sample_data)
        callback2.assert_called_once_with("source1", self.sample_data)


@pytest.mark.asyncio
class TestRealtimeDashboardManager:
    """Comprehensive tests for RealtimeDashboardManager."""
    
    def setup_method(self):
        """Setup test fixtures."""
        dashboard_framework = InteractiveDashboard()
        controller = DashboardController(dashboard_framework)
        self.realtime_manager = RealtimeDashboardManager(controller)
        
        self.sample_data = pd.DataFrame({
            "category": ["A", "B", "C"],
            "value": [10, 20, 30]
        })
    
    async def test_initialization_and_shutdown(self):
        """Test complete initialization and shutdown process."""
        # Initially not initialized
        assert self.realtime_manager.is_initialized is False
        
        # Initialize
        await self.realtime_manager.initialize()
        
        assert self.realtime_manager.is_initialized is True
        assert self.realtime_manager.event_broadcaster.is_running is True
        assert self.realtime_manager.data_stream_manager.is_running is True
        
        # Shutdown
        await self.realtime_manager.shutdown()
        
        assert self.realtime_manager.is_initialized is False
        assert self.realtime_manager.event_broadcaster.is_running is False
        assert self.realtime_manager.data_stream_manager.is_running is False
    
    async def test_client_management_integration(self):
        """Test client management through realtime manager."""
        await self.realtime_manager.initialize()
        
        mock_websocket = AsyncMock()
        
        # Connect client
        await self.realtime_manager.connect_client("client1", mock_websocket, "dashboard1")
        
        # Verify client is connected and subscribed
        assert self.realtime_manager.event_broadcaster.get_client_count() == 1
        
        client = self.realtime_manager.event_broadcaster.clients["client1"]
        assert len(client.subscribed_events) > 0  # Should be auto-subscribed
        
        # Disconnect client
        await self.realtime_manager.disconnect_client("client1")
        
        assert self.realtime_manager.event_broadcaster.get_client_count() == 0
        
        await self.realtime_manager.shutdown()
    
    def test_data_source_integration(self):
        """Test data source management integration."""
        def mock_callback():
            return self.sample_data
        
        # Add data source
        self.realtime_manager.add_data_source("source1", mock_callback, 30)
        
        status = self.realtime_manager.get_status()
        assert status["active_data_sources"] == 1
        assert "source1" in status["data_sources"]
        
        # Remove data source
        self.realtime_manager.remove_data_source("source1")
        
        status = self.realtime_manager.get_status()
        assert status["active_data_sources"] == 0
    
    def test_status_reporting(self):
        """Test status reporting functionality."""
        status = self.realtime_manager.get_status()
        
        assert "initialized" in status
        assert "connected_clients" in status
        assert "active_data_sources" in status
        assert "data_sources" in status
        
        assert isinstance(status["initialized"], bool)
        assert isinstance(status["connected_clients"], int)
        assert isinstance(status["active_data_sources"], int)
        assert isinstance(status["data_sources"], list)
    
    async def test_data_change_propagation(self):
        """Test data change propagation through the system."""
        await self.realtime_manager.initialize()
        
        # Initialize dashboard with data
        self.realtime_manager.dashboard_controller.initialize_dashboard(self.sample_data)
        
        # Mock the data change callback
        with patch.object(self.realtime_manager, '_on_data_change') as mock_callback:
            # Simulate data change
            self.realtime_manager.data_stream_manager._on_data_change("source1", self.sample_data)
            
            # Callback should be triggered
            mock_callback.assert_called_once_with("source1", self.sample_data)
        
        await self.realtime_manager.shutdown()


@pytest.mark.asyncio
class TestDashboardWebSocketHandler:
    """Comprehensive tests for DashboardWebSocketHandler."""
    
    def setup_method(self):
        """Setup test fixtures."""
        dashboard_framework = InteractiveDashboard()
        controller = DashboardController(dashboard_framework)
        realtime_manager = RealtimeDashboardManager(controller)
        self.ws_handler = DashboardWebSocketHandler(realtime_manager)
        
        self.sample_data = pd.DataFrame({
            "category": ["A", "B", "C"],
            "value": [10, 20, 30]
        })
    
    async def test_filter_update_handling(self):
        """Test filter update message handling."""
        # Initialize dashboard
        self.ws_handler.realtime_manager.dashboard_controller.initialize_dashboard(self.sample_data)
        
        # Mock the send response method
        with patch.object(self.ws_handler, '_send_response') as mock_send:
            message = {
                "type": "filter_update",
                "column": "category",
                "values": ["A", "B"]
            }
            
            await self.ws_handler.handle_message("client1", message)
            
            # Verify response was sent
            mock_send.assert_called_once()
            
            # Check response content
            call_args = mock_send.call_args[0]
            client_id = call_args[0]
            response_data = call_args[1]
            
            assert client_id == "client1"
            assert response_data["type"] == "filter_updated"
            assert response_data["column"] == "category"
            assert response_data["values"] == ["A", "B"]
    
    async def test_refresh_request_handling(self):
        """Test refresh request message handling."""
        # Initialize dashboard
        self.ws_handler.realtime_manager.dashboard_controller.initialize_dashboard(self.sample_data)
        
        with patch.object(self.ws_handler, '_send_response') as mock_send:
            message = {"type": "refresh_request"}
            
            await self.ws_handler.handle_message("client1", message)
            
            # Verify response was sent
            mock_send.assert_called_once()
            
            # Check response content
            call_args = mock_send.call_args[0]
            response_data = call_args[1]
            
            assert response_data["type"] == "dashboard_refreshed"
            assert "dashboard_id" in response_data
            assert "timestamp" in response_data
    
    async def test_subscription_handling(self):
        """Test event subscription/unsubscription handling."""
        with patch.object(self.ws_handler, '_send_response') as mock_send:
            # Test subscription
            subscribe_message = {
                "type": "subscribe_events",
                "event_types": ["filter_changed", "data_updated"]
            }
            
            await self.ws_handler.handle_message("client1", subscribe_message)
            
            # Verify subscription response
            mock_send.assert_called()
            response_data = mock_send.call_args[0][1]
            assert response_data["type"] == "subscribed"
            assert response_data["event_types"] == ["filter_changed", "data_updated"]
            
            mock_send.reset_mock()
            
            # Test unsubscription
            unsubscribe_message = {
                "type": "unsubscribe_events",
                "event_types": ["filter_changed"]
            }
            
            await self.ws_handler.handle_message("client1", unsubscribe_message)
            
            # Verify unsubscription response
            mock_send.assert_called()
            response_data = mock_send.call_args[0][1]
            assert response_data["type"] == "unsubscribed"
            assert response_data["event_types"] == ["filter_changed"]
    
    async def test_ping_pong_handling(self):
        """Test ping/pong message handling."""
        with patch.object(self.ws_handler, '_send_response') as mock_send:
            message = {"type": "ping"}
            
            await self.ws_handler.handle_message("client1", message)
            
            # Verify pong response
            mock_send.assert_called_once()
            response_data = mock_send.call_args[0][1]
            
            assert response_data["type"] == "pong"
            assert "timestamp" in response_data
    
    async def test_unknown_message_handling(self):
        """Test handling of unknown message types."""
        with patch.object(self.ws_handler, '_send_error') as mock_error:
            message = {"type": "unknown_message_type"}
            
            await self.ws_handler.handle_message("client1", message)
            
            # Verify error was sent
            mock_error.assert_called_once()
            
            call_args = mock_error.call_args[0]
            client_id = call_args[0]
            error_message = call_args[1]
            
            assert client_id == "client1"
            assert "Unknown message type" in error_message
    
    async def test_malformed_message_handling(self):
        """Test handling of malformed messages."""
        with patch.object(self.ws_handler, '_send_error') as mock_error:
            # Missing required field
            message = {
                "type": "filter_update",
                # Missing "column" field
                "values": ["A", "B"]
            }
            
            await self.ws_handler.handle_message("client1", message)
            
            # Should send error for missing column
            mock_error.assert_called_once()
            error_message = mock_error.call_args[0][1]
            assert "Missing column" in error_message
    
    async def test_error_handling_in_message_processing(self):
        """Test error handling when message processing fails."""
        # Mock a method to raise an exception
        with patch.object(self.ws_handler.realtime_manager.dashboard_controller, 'apply_filter') as mock_apply:
            mock_apply.side_effect = Exception("Processing error")
            
            with patch.object(self.ws_handler, '_send_error') as mock_error:
                message = {
                    "type": "filter_update",
                    "column": "category",
                    "values": ["A", "B"]
                }
                
                await self.ws_handler.handle_message("client1", message)
                
                # Should send error due to processing failure
                mock_error.assert_called_once()
                error_message = mock_error.call_args[0][1]
                assert "Error processing filter_update" in error_message


# Integration tests
@pytest.mark.asyncio
class TestRealtimeIntegration:
    """Integration tests for complete real-time functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.sample_data = pd.DataFrame({
            "category": ["A", "B", "C", "A", "B", "C"],
            "value": [10, 20, 30, 15, 25, 35],
            "date": pd.date_range("2023-01-01", periods=6)
        })
    
    async def test_complete_realtime_workflow(self):
        """Test complete real-time dashboard workflow."""
        # Setup components
        dashboard_framework = InteractiveDashboard()
        controller = DashboardController(dashboard_framework)
        realtime_manager = RealtimeDashboardManager(controller)
        ws_handler = DashboardWebSocketHandler(realtime_manager)
        
        # Initialize dashboard and real-time system
        controller.initialize_dashboard(self.sample_data)
        await realtime_manager.initialize()
        
        # Mock WebSocket connections
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        
        # Connect clients
        await realtime_manager.connect_client("client1", mock_ws1, "dashboard1")
        await realtime_manager.connect_client("client2", mock_ws2, "dashboard1")
        
        # Verify clients are connected
        assert realtime_manager.event_broadcaster.get_client_count() == 2
        
        # Simulate filter update through WebSocket
        filter_message = {
            "type": "filter_update",
            "column": "category",
            "values": ["A", "B"]
        }
        
        with patch.object(ws_handler, '_send_response') as mock_send:
            await ws_handler.handle_message("client1", filter_message)
            
            # Verify response was sent
            mock_send.assert_called_once()
        
        # Add data source for streaming
        def mock_data_source():
            # Return slightly modified data
            modified_data = self.sample_data.copy()
            modified_data["value"] = modified_data["value"] * 2
            return modified_data
        
        realtime_manager.add_data_source("test_source", mock_data_source, 1)
        
        # Wait for potential data updates
        await asyncio.sleep(0.1)
        
        # Verify system status
        status = realtime_manager.get_status()
        assert status["initialized"] is True
        assert status["connected_clients"] == 2
        assert status["active_data_sources"] == 1
        
        # Cleanup
        await realtime_manager.disconnect_client("client1")
        await realtime_manager.disconnect_client("client2")
        await realtime_manager.shutdown()
        
        assert realtime_manager.get_status()["initialized"] is False
    
    async def test_concurrent_client_operations(self):
        """Test concurrent operations with multiple clients."""
        # Setup
        dashboard_framework = InteractiveDashboard()
        controller = DashboardController(dashboard_framework)
        realtime_manager = RealtimeDashboardManager(controller)
        
        controller.initialize_dashboard(self.sample_data)
        await realtime_manager.initialize()
        
        # Connect multiple clients concurrently
        clients = []
        for i in range(5):
            mock_ws = AsyncMock()
            client_id = f"client{i}"
            clients.append((client_id, mock_ws))
        
        # Connect all clients concurrently
        connect_tasks = [
            realtime_manager.connect_client(client_id, ws, "dashboard1")
            for client_id, ws in clients
        ]
        await asyncio.gather(*connect_tasks)
        
        # Verify all clients are connected
        assert realtime_manager.event_broadcaster.get_client_count() == 5
        
        # Disconnect all clients concurrently
        disconnect_tasks = [
            realtime_manager.disconnect_client(client_id)
            for client_id, _ in clients
        ]
        await asyncio.gather(*disconnect_tasks)
        
        # Verify all clients are disconnected
        assert realtime_manager.event_broadcaster.get_client_count() == 0
        
        await realtime_manager.shutdown()
    
    async def test_data_streaming_with_client_updates(self):
        """Test data streaming with real client updates."""
        # Setup
        dashboard_framework = InteractiveDashboard()
        controller = DashboardController(dashboard_framework)
        realtime_manager = RealtimeDashboardManager(controller)
        
        controller.initialize_dashboard(self.sample_data)
        await realtime_manager.initialize()
        
        # Connect client
        mock_websocket = AsyncMock()
        await realtime_manager.connect_client("client1", mock_websocket, "dashboard1")
        
        # Track messages sent to client
        sent_messages = []
        
        def capture_send(message):
            sent_messages.append(json.loads(message))
            return AsyncMock()
        
        mock_websocket.send.side_effect = capture_send
        
        # Add data source that changes
        call_count = 0
        
        def changing_data_source():
            nonlocal call_count
            call_count += 1
            modified_data = self.sample_data.copy()
            modified_data["value"] = modified_data["value"] + call_count
            return modified_data
        
        realtime_manager.add_data_source("changing_source", changing_data_source, 0.1)
        
        # Wait for some updates
        await asyncio.sleep(0.5)
        
        # Should have received multiple messages
        assert len(sent_messages) > 1  # At least connection + some updates
        
        # Check that we received connection and update messages
        message_types = [msg.get("message_type") for msg in sent_messages]
        assert "connection_established" in message_types
        
        await realtime_manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])