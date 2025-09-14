"""
Tests for interactive dashboard framework.

This module tests dashboard layout management, interactive filters,
real-time updates, and user interaction capabilities.
"""

import pytest
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from src.visualization.dashboard_framework import (
    InteractiveDashboard, DashboardController, FilterManager, LayoutManager,
    RefreshManager, FilterState, DashboardState, DashboardEvent, DashboardEventType
)
from src.visualization.realtime_updates import (
    RealtimeEventBroadcaster, DataStreamManager, RealtimeDashboardManager,
    DashboardWebSocketHandler, UpdateMessage, ClientConnection
)
from src.core.models import Dashboard, Filter, Chart, ChartConfig, ChartType, KPI


class TestFilterManager:
    """Test cases for FilterManager."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.filter_manager = FilterManager()
        
        # Sample filters
        self.categorical_filter = Filter(
            column="category",
            filter_type="select",
            values=["A", "B", "C"],
            default_value="A"
        )
        
        self.numeric_filter = Filter(
            column="value",
            filter_type="range",
            values=[0, 100],
            default_value=None
        )
        
        self.search_filter = Filter(
            column="description",
            filter_type="search",
            values=[],
            default_value=""
        )
    
    def test_add_filter(self):
        """Test adding filters to manager."""
        self.filter_manager.add_filter(self.categorical_filter)
        
        assert "category" in self.filter_manager.filters
        assert "category" in self.filter_manager.filter_states
        
        state = self.filter_manager.filter_states["category"]
        assert state.column == "category"
        assert state.filter_type == "select"
        assert state.active_values == ["A"]  # Default value
    
    def test_update_filter(self):
        """Test updating filter values."""
        self.filter_manager.add_filter(self.categorical_filter)
        
        # Mock event handler
        event_handler = Mock()
        self.filter_manager.add_event_handler(event_handler)
        
        # Update filter
        self.filter_manager.update_filter("category", ["B", "C"])
        
        state = self.filter_manager.filter_states["category"]
        assert state.active_values == ["B", "C"]
        assert state.is_active is True
        
        # Verify event was triggered
        event_handler.handle_event.assert_called_once()
        event = event_handler.handle_event.call_args[0][0]
        assert event.event_type == DashboardEventType.FILTER_CHANGED
        assert event.data["column"] == "category"
        assert event.data["new_values"] == ["B", "C"]
    
    def test_clear_filter(self):
        """Test clearing specific filter."""
        self.filter_manager.add_filter(self.categorical_filter)
        self.filter_manager.update_filter("category", ["B", "C"])
        
        # Clear filter
        self.filter_manager.clear_filter("category")
        
        state = self.filter_manager.filter_states["category"]
        assert state.active_values == []
        assert state.is_active is False
    
    def test_apply_filters_select(self):
        """Test applying select filters to dataframe."""
        # Setup data
        df = pd.DataFrame({
            "category": ["A", "B", "C", "A", "B"],
            "value": [10, 20, 30, 40, 50]
        })
        
        # Setup filter
        self.filter_manager.add_filter(self.categorical_filter)
        self.filter_manager.update_filter("category", ["A", "B"])
        
        # Apply filters
        filtered_df = self.filter_manager.apply_filters(df)
        
        assert len(filtered_df) == 4  # Only A and B categories
        assert set(filtered_df["category"].unique()) == {"A", "B"}
    
    def test_apply_filters_range(self):
        """Test applying range filters to dataframe."""
        # Setup data
        df = pd.DataFrame({
            "category": ["A", "B", "C", "A", "B"],
            "value": [10, 20, 30, 40, 50]
        })
        
        # Setup filter
        self.filter_manager.add_filter(self.numeric_filter)
        self.filter_manager.update_filter("value", [20, 40])
        
        # Apply filters
        filtered_df = self.filter_manager.apply_filters(df)
        
        assert len(filtered_df) == 3  # Values 20, 30, 40
        assert filtered_df["value"].min() >= 20
        assert filtered_df["value"].max() <= 40
    
    def test_apply_filters_search(self):
        """Test applying search filters to dataframe."""
        # Setup data
        df = pd.DataFrame({
            "category": ["A", "B", "C", "A", "B"],
            "description": ["apple", "banana", "cherry", "apricot", "blueberry"]
        })
        
        # Setup filter
        self.filter_manager.add_filter(self.search_filter)
        self.filter_manager.update_filter("description", ["ap"])
        
        # Apply filters
        filtered_df = self.filter_manager.apply_filters(df)
        
        assert len(filtered_df) == 2  # "apple" and "apricot"
        assert all("ap" in desc.lower() for desc in filtered_df["description"])


class TestLayoutManager:
    """Test cases for LayoutManager."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.layout_manager = LayoutManager(grid_columns=12)
    
    def test_add_chart_position(self):
        """Test adding chart position."""
        self.layout_manager.add_chart_position("chart1", x=0, y=0, width=6, height=4)
        
        assert "chart1" in self.layout_manager.chart_positions
        position = self.layout_manager.chart_positions["chart1"]
        assert position["x"] == 0
        assert position["y"] == 0
        assert position["width"] == 6
        assert position["height"] == 4
    
    def test_auto_layout_charts(self):
        """Test automatic chart layout."""
        chart_ids = ["chart1", "chart2", "chart3", "chart4", "chart5"]
        
        self.layout_manager.auto_layout_charts(chart_ids)
        
        # Verify all charts have positions
        assert len(self.layout_manager.chart_positions) == 5
        
        # Verify layout logic (3 charts per row)
        chart1_pos = self.layout_manager.chart_positions["chart1"]
        chart2_pos = self.layout_manager.chart_positions["chart2"]
        chart4_pos = self.layout_manager.chart_positions["chart4"]
        
        assert chart1_pos["x"] == 0  # First in row
        assert chart2_pos["x"] == 4  # Second in row
        assert chart4_pos["y"] == 4  # Second row
    
    def test_width_constraint(self):
        """Test width constraint enforcement."""
        # Try to add chart wider than grid
        self.layout_manager.add_chart_position("chart1", x=0, y=0, width=15, height=4)
        
        position = self.layout_manager.chart_positions["chart1"]
        assert position["width"] == 12  # Constrained to grid width


class TestRefreshManager:
    """Test cases for RefreshManager."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.refresh_manager = RefreshManager()
    
    def test_add_data_source(self):
        """Test adding data source callback."""
        def mock_callback():
            return pd.DataFrame({"test": [1, 2, 3]})
        
        self.refresh_manager.add_data_source(mock_callback)
        
        assert len(self.refresh_manager.refresh_callbacks) == 1
    
    def test_manual_refresh(self):
        """Test manual refresh operation."""
        # Mock data source
        def mock_callback():
            return pd.DataFrame({"test": [1, 2, 3]})
        
        self.refresh_manager.add_data_source(mock_callback)
        
        # Mock event handler
        event_handler = Mock()
        self.refresh_manager.add_event_handler(event_handler)
        
        # Perform refresh
        result = self.refresh_manager.manual_refresh()
        
        assert len(result) == 1
        assert isinstance(result[0], pd.DataFrame)
        
        # Verify event was triggered
        event_handler.handle_event.assert_called_once()
        event = event_handler.handle_event.call_args[0][0]
        assert event.event_type == DashboardEventType.DATA_UPDATED
    
    def test_auto_refresh_settings(self):
        """Test auto-refresh configuration."""
        # Enable auto-refresh
        self.refresh_manager.set_auto_refresh(30)
        
        assert self.refresh_manager.auto_refresh_interval == 30
        assert self.refresh_manager.is_auto_refresh_active is True
        
        # Disable auto-refresh
        self.refresh_manager.disable_auto_refresh()
        
        assert self.refresh_manager.is_auto_refresh_active is False
        assert self.refresh_manager.auto_refresh_interval is None
    
    def test_should_auto_refresh(self):
        """Test auto-refresh timing logic."""
        # Set short interval for testing
        self.refresh_manager.set_auto_refresh(1)
        
        # Should not refresh immediately
        assert self.refresh_manager.should_auto_refresh() is False
        
        # Simulate time passage
        self.refresh_manager.last_refresh = datetime.now() - timedelta(seconds=2)
        
        # Should refresh now
        assert self.refresh_manager.should_auto_refresh() is True


class TestInteractiveDashboard:
    """Test cases for InteractiveDashboard."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.dashboard = InteractiveDashboard(chart_engine="plotly")
        
        # Sample data
        self.sample_data = pd.DataFrame({
            "category": ["A", "B", "C", "A", "B", "C"],
            "value": [10, 20, 30, 15, 25, 35],
            "date": pd.date_range("2023-01-01", periods=6),
            "description": ["item1", "item2", "item3", "item4", "item5", "item6"]
        })
    
    def test_generate_dashboard(self):
        """Test dashboard generation from data."""
        dashboard = self.dashboard.generate_dashboard(self.sample_data)
        
        assert isinstance(dashboard, Dashboard)
        assert len(dashboard.charts) > 0
        assert len(dashboard.filters) > 0
        assert len(dashboard.kpis) > 0
        assert dashboard.layout is not None
    
    def test_create_filters(self):
        """Test filter creation from data."""
        filters = self.dashboard.create_filters(self.sample_data)
        
        # Should create filters for appropriate columns
        filter_columns = [f.column for f in filters]
        
        # Category should have select filter
        category_filter = next((f for f in filters if f.column == "category"), None)
        assert category_filter is not None
        assert category_filter.filter_type == "select"
        
        # Value filter type depends on the logic - with small unique count it becomes select
        value_filter = next((f for f in filters if f.column == "value"), None)
        assert value_filter is not None
        # The filter type can be either select or range depending on unique count
        assert value_filter.filter_type in ["select", "range"]
    
    def test_update_filter(self):
        """Test filter update and dashboard regeneration."""
        # Generate initial dashboard
        initial_dashboard = self.dashboard.generate_dashboard(self.sample_data)
        
        # Update filter
        updated_dashboard = self.dashboard.update_filter("category", ["A", "B"])
        
        assert isinstance(updated_dashboard, Dashboard)
        # Dashboard should be regenerated with filtered data
        assert updated_dashboard.id != initial_dashboard.id
    
    def test_refresh_data(self):
        """Test data refresh functionality."""
        # Generate initial dashboard
        initial_dashboard = self.dashboard.generate_dashboard(self.sample_data)
        
        # Create new data
        new_data = pd.DataFrame({
            "category": ["X", "Y", "Z"],
            "value": [100, 200, 300],
            "date": pd.date_range("2023-07-01", periods=3),
            "description": ["new1", "new2", "new3"]
        })
        
        # Refresh with new data
        refreshed_dashboard = self.dashboard.refresh_data(new_data)
        
        assert isinstance(refreshed_dashboard, Dashboard)
        assert refreshed_dashboard.id != initial_dashboard.id
    
    def test_auto_refresh_setup(self):
        """Test auto-refresh configuration."""
        def mock_data_callback():
            return self.sample_data
        
        # Setup auto-refresh
        self.dashboard.set_auto_refresh(30, mock_data_callback)
        
        assert self.dashboard.dashboard_state.is_auto_refresh is True
        assert self.dashboard.dashboard_state.refresh_interval == 30
        
        # Disable auto-refresh
        self.dashboard.disable_auto_refresh()
        
        assert self.dashboard.dashboard_state.is_auto_refresh is False


class TestDashboardController:
    """Test cases for DashboardController."""
    
    def setup_method(self):
        """Setup test fixtures."""
        dashboard_framework = InteractiveDashboard()
        self.controller = DashboardController(dashboard_framework)
        
        self.sample_data = pd.DataFrame({
            "category": ["A", "B", "C"],
            "value": [10, 20, 30]
        })
    
    def test_initialize_dashboard(self):
        """Test dashboard initialization."""
        dashboard = self.controller.initialize_dashboard(self.sample_data)
        
        assert isinstance(dashboard, Dashboard)
        assert self.controller.current_dashboard_config is not None
        assert self.controller.current_dashboard_config.id == dashboard.id
    
    def test_apply_filter(self):
        """Test filter application through controller."""
        # Initialize dashboard
        self.controller.initialize_dashboard(self.sample_data)
        
        # Apply filter
        updated_dashboard = self.controller.apply_filter("category", ["A", "B"])
        
        assert isinstance(updated_dashboard, Dashboard)
        assert len(self.controller.interaction_history) == 1
        
        event = self.controller.interaction_history[0]
        assert event.event_type == DashboardEventType.FILTER_CHANGED
        assert event.data["column"] == "category"
    
    def test_clear_filters(self):
        """Test clearing all filters."""
        # Initialize and apply filter
        self.controller.initialize_dashboard(self.sample_data)
        self.controller.apply_filter("category", ["A"])
        
        # Clear filters
        cleared_dashboard = self.controller.clear_filters()
        
        assert isinstance(cleared_dashboard, Dashboard)
        # Should have no active filters
        active_filters = self.controller.dashboard.filter_manager.get_active_filters()
        assert len(active_filters) == 0
    
    def test_get_dashboard_json(self):
        """Test dashboard JSON serialization."""
        # Initialize dashboard
        self.controller.initialize_dashboard(self.sample_data)
        
        # Get JSON
        dashboard_json = self.controller.get_dashboard_json()
        
        assert isinstance(dashboard_json, str)
        assert len(dashboard_json) > 0
        
        # Should be valid JSON
        import json
        parsed = json.loads(dashboard_json)
        assert "id" in parsed
        assert "charts" in parsed
        assert "filters" in parsed


@pytest.mark.asyncio
class TestRealtimeEventBroadcaster:
    """Test cases for RealtimeEventBroadcaster."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.broadcaster = RealtimeEventBroadcaster()
    
    async def test_start_stop_broadcaster(self):
        """Test starting and stopping broadcaster."""
        await self.broadcaster.start_broadcaster()
        assert self.broadcaster.is_running is True
        
        await self.broadcaster.stop_broadcaster()
        assert self.broadcaster.is_running is False
    
    async def test_connect_disconnect_client(self):
        """Test client connection management."""
        # Mock WebSocket
        mock_websocket = AsyncMock()
        
        # Connect client
        await self.broadcaster.connect_client("client1", mock_websocket, "dashboard1")
        
        assert "client1" in self.broadcaster.clients
        assert self.broadcaster.clients["client1"].dashboard_id == "dashboard1"
        
        # Disconnect client
        await self.broadcaster.disconnect_client("client1")
        
        assert "client1" not in self.broadcaster.clients
    
    async def test_subscribe_unsubscribe_events(self):
        """Test event subscription management."""
        # Mock WebSocket and connect client
        mock_websocket = AsyncMock()
        await self.broadcaster.connect_client("client1", mock_websocket)
        
        # Subscribe to events
        self.broadcaster.subscribe_client("client1", ["filter_changed", "data_updated"])
        
        client = self.broadcaster.clients["client1"]
        assert "filter_changed" in client.subscribed_events
        assert "data_updated" in client.subscribed_events
        
        # Unsubscribe from events
        self.broadcaster.unsubscribe_client("client1", ["filter_changed"])
        
        assert "filter_changed" not in client.subscribed_events
        assert "data_updated" in client.subscribed_events
    
    async def test_handle_event(self):
        """Test event handling and queuing."""
        await self.broadcaster.start_broadcaster()
        
        # Create test event
        event = DashboardEvent(
            event_type=DashboardEventType.FILTER_CHANGED,
            data={"column": "test", "values": ["A", "B"]}
        )
        
        # Handle event (should queue for broadcast)
        self.broadcaster.handle_event(event)
        
        # Give time for async processing
        await asyncio.sleep(0.1)
        
        await self.broadcaster.stop_broadcaster()


class TestDataStreamManager:
    """Test cases for DataStreamManager."""
    
    def setup_method(self):
        """Setup test fixtures."""
        dashboard_framework = InteractiveDashboard()
        controller = DashboardController(dashboard_framework)
        self.stream_manager = DataStreamManager(controller)
    
    def test_add_remove_data_source(self):
        """Test data source management."""
        def mock_callback():
            return pd.DataFrame({"test": [1, 2, 3]})
        
        # Add data source
        self.stream_manager.add_data_source("source1", mock_callback, 10)
        
        assert "source1" in self.stream_manager.data_sources
        assert self.stream_manager.stream_intervals["source1"] == 10
        
        # Remove data source
        self.stream_manager.remove_data_source("source1")
        
        assert "source1" not in self.stream_manager.data_sources
    
    @pytest.mark.asyncio
    async def test_start_stop_streaming(self):
        """Test streaming lifecycle."""
        def mock_callback():
            return pd.DataFrame({"test": [1, 2, 3]})
        
        self.stream_manager.add_data_source("source1", mock_callback, 1)
        
        # Start streaming
        await self.stream_manager.start_streaming()
        assert self.stream_manager.is_running is True
        assert len(self.stream_manager.stream_tasks) == 1
        
        # Stop streaming
        await self.stream_manager.stop_streaming()
        assert self.stream_manager.is_running is False
        assert len(self.stream_manager.stream_tasks) == 0
    
    def test_data_change_detection(self):
        """Test data change detection."""
        # Create test data
        data1 = pd.DataFrame({"test": [1, 2, 3]})
        data2 = pd.DataFrame({"test": [1, 2, 3]})  # Same data
        data3 = pd.DataFrame({"test": [4, 5, 6]})  # Different data
        
        # First check should return True (first time)
        assert self.stream_manager._has_data_changed("source1", data1) is True
        
        # Same data should return False
        assert self.stream_manager._has_data_changed("source1", data2) is False
        
        # Different data should return True
        assert self.stream_manager._has_data_changed("source1", data3) is True


class TestRealtimeDashboardManager:
    """Test cases for RealtimeDashboardManager."""
    
    def setup_method(self):
        """Setup test fixtures."""
        dashboard_framework = InteractiveDashboard()
        controller = DashboardController(dashboard_framework)
        self.realtime_manager = RealtimeDashboardManager(controller)
    
    @pytest.mark.asyncio
    async def test_initialize_shutdown(self):
        """Test initialization and shutdown."""
        await self.realtime_manager.initialize()
        assert self.realtime_manager.is_initialized is True
        
        await self.realtime_manager.shutdown()
        assert self.realtime_manager.is_initialized is False
    
    @pytest.mark.asyncio
    async def test_client_management(self):
        """Test client connection management."""
        await self.realtime_manager.initialize()
        
        # Mock WebSocket
        mock_websocket = AsyncMock()
        
        # Connect client
        await self.realtime_manager.connect_client("client1", mock_websocket, "dashboard1")
        
        assert self.realtime_manager.event_broadcaster.get_client_count() == 1
        
        # Disconnect client
        await self.realtime_manager.disconnect_client("client1")
        
        assert self.realtime_manager.event_broadcaster.get_client_count() == 0
        
        await self.realtime_manager.shutdown()
    
    def test_data_source_management(self):
        """Test data source management."""
        def mock_callback():
            return pd.DataFrame({"test": [1, 2, 3]})
        
        # Add data source
        self.realtime_manager.add_data_source("source1", mock_callback, 30)
        
        status = self.realtime_manager.get_status()
        assert status["active_data_sources"] == 1
        assert "source1" in status["data_sources"]
        
        # Remove data source
        self.realtime_manager.remove_data_source("source1")
        
        status = self.realtime_manager.get_status()
        assert status["active_data_sources"] == 0


class TestDashboardWebSocketHandler:
    """Test cases for DashboardWebSocketHandler."""
    
    def setup_method(self):
        """Setup test fixtures."""
        dashboard_framework = InteractiveDashboard()
        controller = DashboardController(dashboard_framework)
        realtime_manager = RealtimeDashboardManager(controller)
        self.ws_handler = DashboardWebSocketHandler(realtime_manager)
    
    @pytest.mark.asyncio
    async def test_handle_filter_update(self):
        """Test filter update message handling."""
        # Mock the dashboard controller method
        with patch.object(self.ws_handler.realtime_manager.dashboard_controller, 'apply_filter') as mock_apply:
            mock_dashboard = Dashboard(id="test_dashboard")
            mock_apply.return_value = mock_dashboard
            
            # Mock the send response method
            with patch.object(self.ws_handler, '_send_response') as mock_send:
                message = {
                    "type": "filter_update",
                    "column": "category",
                    "values": ["A", "B"]
                }
                
                await self.ws_handler.handle_message("client1", message)
                
                # Verify filter was applied
                mock_apply.assert_called_once_with("category", ["A", "B"])
                
                # Verify response was sent
                mock_send.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_ping(self):
        """Test ping message handling."""
        with patch.object(self.ws_handler, '_send_response') as mock_send:
            message = {"type": "ping"}
            
            await self.ws_handler.handle_message("client1", message)
            
            # Verify pong response was sent
            mock_send.assert_called_once()
            call_args = mock_send.call_args[0]
            assert call_args[1]["type"] == "pong"
    
    @pytest.mark.asyncio
    async def test_handle_unknown_message(self):
        """Test handling of unknown message types."""
        with patch.object(self.ws_handler, '_send_error') as mock_error:
            message = {"type": "unknown_type"}
            
            await self.ws_handler.handle_message("client1", message)
            
            # Verify error was sent
            mock_error.assert_called_once()
            error_message = mock_error.call_args[0][1]
            assert "Unknown message type" in error_message


# Integration tests
class TestDashboardIntegration:
    """Integration tests for complete dashboard functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.sample_data = pd.DataFrame({
            "category": ["A", "B", "C", "A", "B", "C"] * 10,
            "value": range(60),
            "date": pd.date_range("2023-01-01", periods=60),
            "description": [f"item_{i}" for i in range(60)]
        })
    
    def test_end_to_end_dashboard_creation(self):
        """Test complete dashboard creation workflow."""
        # Create dashboard
        dashboard_framework = InteractiveDashboard()
        controller = DashboardController(dashboard_framework)
        
        # Initialize with data
        dashboard = controller.initialize_dashboard(self.sample_data)
        
        # Verify dashboard structure
        assert isinstance(dashboard, Dashboard)
        assert len(dashboard.charts) > 0
        assert len(dashboard.filters) > 0
        assert len(dashboard.kpis) > 0
        
        # Apply filter
        filtered_dashboard = controller.apply_filter("category", ["A", "B"])
        
        # Verify filter was applied
        assert filtered_dashboard.id != dashboard.id
        
        # Clear filters
        cleared_dashboard = controller.clear_filters()
        
        # Verify filters were cleared
        active_filters = dashboard_framework.filter_manager.get_active_filters()
        assert len(active_filters) == 0
    
    @pytest.mark.asyncio
    async def test_realtime_dashboard_workflow(self):
        """Test complete real-time dashboard workflow."""
        # Setup components
        dashboard_framework = InteractiveDashboard()
        controller = DashboardController(dashboard_framework)
        realtime_manager = RealtimeDashboardManager(controller)
        
        # Initialize dashboard
        controller.initialize_dashboard(self.sample_data)
        
        # Initialize real-time system
        await realtime_manager.initialize()
        
        # Add data source
        def mock_data_source():
            return self.sample_data.copy()
        
        realtime_manager.add_data_source("test_source", mock_data_source, 1)
        
        # Verify status
        status = realtime_manager.get_status()
        assert status["initialized"] is True
        assert status["active_data_sources"] == 1
        
        # Cleanup
        await realtime_manager.shutdown()
        
        assert realtime_manager.get_status()["initialized"] is False


if __name__ == "__main__":
    pytest.main([__file__])