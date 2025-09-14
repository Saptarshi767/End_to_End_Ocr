"""
Tests for advanced dashboard layout management system.

This module tests layout templates, responsive design, drag-and-drop positioning,
and layout optimization functionality.
"""

import pytest
from unittest.mock import Mock, patch

from src.visualization.dashboard_layouts import (
    AdvancedLayoutManager, LayoutType, ResponsiveBreakpoint, 
    LayoutConstraints, ResponsiveLayout, LayoutTemplate
)
from src.core.models import DashboardLayout


class TestAdvancedLayoutManager:
    """Test cases for AdvancedLayoutManager."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.layout_manager = AdvancedLayoutManager(grid_columns=12)
    
    def test_initialization(self):
        """Test layout manager initialization."""
        assert self.layout_manager.grid_columns == 12
        assert self.layout_manager.layout_type == LayoutType.GRID
        assert len(self.layout_manager.layout_templates) > 0
    
    def test_add_widget_position(self):
        """Test adding widget position."""
        result = self.layout_manager.add_widget_position(
            "widget1", x=0, y=0, width=6, height=4
        )
        
        assert result is True
        assert "widget1" in self.layout_manager.widget_positions
        
        position = self.layout_manager.widget_positions["widget1"]
        assert position["x"] == 0
        assert position["y"] == 0
        assert position["width"] == 6
        assert position["height"] == 4
    
    def test_add_widget_with_constraints(self):
        """Test adding widget with layout constraints."""
        constraints = LayoutConstraints(
            min_width=2,
            max_width=8,
            min_height=2,
            max_height=6,
            aspect_ratio=2.0
        )
        
        result = self.layout_manager.add_widget_position(
            "widget1", x=0, y=0, width=10, height=3,
            constraints=constraints
        )
        
        assert result is True
        position = self.layout_manager.widget_positions["widget1"]
        
        # Width should be constrained to max_width
        assert position["width"] == 8
        # Height should be adjusted for aspect ratio
        assert position["height"] == 4  # 8 / 2.0
    
    def test_widget_position_overflow(self):
        """Test widget position when it overflows grid."""
        result = self.layout_manager.add_widget_position(
            "widget1", x=10, y=0, width=6, height=4
        )
        
        assert result is True
        position = self.layout_manager.widget_positions["widget1"]
        
        # X should be adjusted to fit within grid
        assert position["x"] == 6  # 12 - 6 = 6
        assert position["width"] == 6
    
    def test_move_widget(self):
        """Test moving widget to new position."""
        # Add widget first
        self.layout_manager.add_widget_position("widget1", 0, 0, 4, 3)
        
        # Move widget
        result = self.layout_manager.move_widget("widget1", 6, 2)
        
        assert result is True
        position = self.layout_manager.widget_positions["widget1"]
        assert position["x"] == 6
        assert position["y"] == 2
    
    def test_move_widget_invalid_position(self):
        """Test moving widget to invalid position."""
        self.layout_manager.add_widget_position("widget1", 0, 0, 4, 3)
        
        # Try to move outside grid
        result = self.layout_manager.move_widget("widget1", 15, 0)
        
        assert result is False
        # Position should remain unchanged
        position = self.layout_manager.widget_positions["widget1"]
        assert position["x"] == 0
    
    def test_resize_widget(self):
        """Test resizing widget."""
        self.layout_manager.add_widget_position("widget1", 0, 0, 4, 3)
        
        result = self.layout_manager.resize_widget("widget1", 6, 4)
        
        assert result is True
        position = self.layout_manager.widget_positions["widget1"]
        assert position["width"] == 6
        assert position["height"] == 4
    
    def test_resize_widget_with_constraints(self):
        """Test resizing widget with constraints."""
        constraints = LayoutConstraints(
            min_width=2,
            max_width=8,
            aspect_ratio=2.0
        )
        
        self.layout_manager.add_widget_position(
            "widget1", 0, 0, 4, 2, constraints=constraints
        )
        
        # Try to resize beyond constraints
        result = self.layout_manager.resize_widget("widget1", 10, 3)
        
        assert result is True
        position = self.layout_manager.widget_positions["widget1"]
        
        # Width should be constrained
        assert position["width"] == 8
        # Height should maintain aspect ratio
        assert position["height"] == 4  # 8 / 2.0
    
    def test_auto_layout_balanced(self):
        """Test balanced auto-layout strategy."""
        widget_ids = ["widget1", "widget2", "widget3", "widget4", "widget5"]
        
        self.layout_manager.auto_layout_widgets(widget_ids, "balanced")
        
        # Check that all widgets have positions
        assert len(self.layout_manager.widget_positions) == 5
        
        # Check first row (3 widgets)
        assert self.layout_manager.widget_positions["widget1"]["x"] == 0
        assert self.layout_manager.widget_positions["widget2"]["x"] == 4
        assert self.layout_manager.widget_positions["widget3"]["x"] == 8
        
        # Check second row
        assert self.layout_manager.widget_positions["widget4"]["y"] == 4
        assert self.layout_manager.widget_positions["widget5"]["y"] == 4
    
    def test_auto_layout_compact(self):
        """Test compact auto-layout strategy."""
        widget_ids = ["widget1", "widget2", "widget3"]
        
        self.layout_manager.auto_layout_widgets(widget_ids, "compact")
        
        # Check that widgets are placed compactly
        positions = self.layout_manager.widget_positions
        
        # First widget should be at origin
        assert positions["widget1"]["x"] == 0
        assert positions["widget1"]["y"] == 0
        
        # Second widget should be adjacent
        assert positions["widget2"]["x"] == 4  # After first widget
        assert positions["widget2"]["y"] == 0
    
    def test_auto_layout_grid(self):
        """Test grid auto-layout strategy."""
        widget_ids = ["widget1", "widget2", "widget3", "widget4"]
        
        self.layout_manager.auto_layout_widgets(widget_ids, "grid")
        
        positions = self.layout_manager.widget_positions
        
        # Check grid arrangement (3 columns)
        assert positions["widget1"]["x"] == 0
        assert positions["widget2"]["x"] == 4
        assert positions["widget3"]["x"] == 8
        assert positions["widget4"]["x"] == 0  # Second row
        assert positions["widget4"]["y"] == 4
    
    def test_overlap_detection(self):
        """Test overlap detection between widgets."""
        # Add first widget
        self.layout_manager.add_widget_position("widget1", 0, 0, 4, 3)
        
        # Try to add overlapping widget
        has_overlap = self.layout_manager._has_overlap("widget2", 2, 1, 4, 3)
        
        assert has_overlap is True
        
        # Try non-overlapping position
        has_overlap = self.layout_manager._has_overlap("widget2", 6, 0, 4, 3)
        
        assert has_overlap is False
    
    def test_find_available_position(self):
        """Test finding available position for widget."""
        # Fill some positions
        self.layout_manager.add_widget_position("widget1", 0, 0, 6, 3)
        self.layout_manager.add_widget_position("widget2", 6, 0, 6, 3)
        
        # Find position for new widget
        position = self.layout_manager._find_available_position(4, 2)
        
        assert position is not None
        x, y = position
        
        # Should find position below existing widgets
        assert y >= 3
    
    def test_apply_template(self):
        """Test applying layout template."""
        widget_ids = ["widget1", "widget2", "widget3", "widget4"]
        
        result = self.layout_manager.apply_template("executive", widget_ids)
        
        assert result is True
        assert len(self.layout_manager.widget_positions) == 4
        
        # Check that positions match template
        template = self.layout_manager.layout_templates["executive"]
        for i, widget_id in enumerate(widget_ids):
            if i < len(template.widget_slots):
                slot = template.widget_slots[i]
                position = self.layout_manager.widget_positions[widget_id]
                assert position["x"] == slot["x"]
                assert position["y"] == slot["y"]
    
    def test_apply_invalid_template(self):
        """Test applying non-existent template."""
        result = self.layout_manager.apply_template("nonexistent", ["widget1"])
        
        assert result is False
    
    def test_responsive_layout(self):
        """Test responsive layout functionality."""
        # Create responsive layout for mobile
        mobile_layout = ResponsiveLayout(
            breakpoint=ResponsiveBreakpoint.SM,
            grid_columns=6,
            widget_positions={"widget1": {"x": 0, "y": 0, "width": 6, "height": 4}},
            hidden_widgets=["widget2"]
        )
        
        self.layout_manager.set_responsive_layout(ResponsiveBreakpoint.SM, mobile_layout)
        
        # Get layout for mobile breakpoint
        layout = self.layout_manager.get_layout_for_breakpoint(ResponsiveBreakpoint.SM)
        
        assert layout["grid_columns"] == 6
        assert "widget1" in layout["widget_positions"]
        assert "widget2" in layout["hidden_widgets"]
    
    def test_optimize_layout(self):
        """Test layout optimization."""
        # Add widgets with gaps
        self.layout_manager.add_widget_position("widget1", 0, 0, 4, 3)
        self.layout_manager.add_widget_position("widget2", 6, 0, 4, 3)  # Gap at x=4-5
        self.layout_manager.add_widget_position("widget3", 0, 5, 4, 3)  # Gap at y=3-4
        
        # Optimize layout
        self.layout_manager.optimize_layout()
        
        # Check that gaps are minimized
        positions = self.layout_manager.widget_positions
        
        # Widgets in same row should be adjacent
        assert positions["widget1"]["x"] == 0
        assert positions["widget2"]["x"] == 4  # Should be moved to close gap
    
    def test_export_import_layout(self):
        """Test layout export and import."""
        # Setup layout
        self.layout_manager.add_widget_position("widget1", 0, 0, 4, 3)
        self.layout_manager.add_widget_position("widget2", 4, 0, 4, 3)
        
        constraints = LayoutConstraints(min_width=2, max_width=8)
        self.layout_manager.widget_constraints["widget1"] = constraints
        
        # Export layout
        exported = self.layout_manager.export_layout()
        
        assert "grid_columns" in exported
        assert "widget_positions" in exported
        assert "widget_constraints" in exported
        
        # Create new manager and import
        new_manager = AdvancedLayoutManager()
        result = new_manager.import_layout(exported)
        
        assert result is True
        assert new_manager.grid_columns == self.layout_manager.grid_columns
        assert len(new_manager.widget_positions) == 2
        assert "widget1" in new_manager.widget_constraints
    
    def test_get_available_templates(self):
        """Test getting available layout templates."""
        templates = self.layout_manager.get_available_templates()
        
        assert len(templates) > 0
        
        template_ids = [t["id"] for t in templates]
        assert "executive" in template_ids
        assert "analytical" in template_ids
        assert "operational" in template_ids
        
        # Check template structure
        for template in templates:
            assert "id" in template
            assert "name" in template
            assert "description" in template
            assert "layout_type" in template
    
    def test_add_custom_template(self):
        """Test adding custom layout template."""
        custom_template = LayoutTemplate(
            template_id="custom",
            name="Custom Template",
            description="A custom layout template",
            layout_type=LayoutType.GRID,
            widget_slots=[
                {"type": "chart", "x": 0, "y": 0, "width": 12, "height": 6}
            ]
        )
        
        self.layout_manager.add_custom_template(custom_template)
        
        templates = self.layout_manager.get_available_templates()
        template_ids = [t["id"] for t in templates]
        assert "custom" in template_ids
    
    def test_get_layout_config(self):
        """Test getting layout configuration."""
        self.layout_manager.add_widget_position("widget1", 0, 0, 4, 3)
        
        config = self.layout_manager.get_layout_config()
        
        assert isinstance(config, DashboardLayout)
        assert config.grid_columns == 12
        assert "widget1" in config.chart_positions


class TestLayoutConstraints:
    """Test cases for LayoutConstraints."""
    
    def test_constraint_creation(self):
        """Test creating layout constraints."""
        constraints = LayoutConstraints(
            min_width=2,
            max_width=8,
            min_height=1,
            max_height=6,
            aspect_ratio=2.0,
            fixed_position=True
        )
        
        assert constraints.min_width == 2
        assert constraints.max_width == 8
        assert constraints.aspect_ratio == 2.0
        assert constraints.fixed_position is True
    
    def test_default_constraints(self):
        """Test default constraint values."""
        constraints = LayoutConstraints()
        
        assert constraints.min_width == 1
        assert constraints.max_width == 12
        assert constraints.min_height == 1
        assert constraints.max_height == 20
        assert constraints.aspect_ratio is None
        assert constraints.fixed_position is False


class TestResponsiveLayout:
    """Test cases for ResponsiveLayout."""
    
    def test_responsive_layout_creation(self):
        """Test creating responsive layout."""
        layout = ResponsiveLayout(
            breakpoint=ResponsiveBreakpoint.MD,
            grid_columns=8,
            widget_positions={"widget1": {"x": 0, "y": 0, "width": 4, "height": 3}},
            hidden_widgets=["widget2", "widget3"]
        )
        
        assert layout.breakpoint == ResponsiveBreakpoint.MD
        assert layout.grid_columns == 8
        assert len(layout.widget_positions) == 1
        assert len(layout.hidden_widgets) == 2


class TestLayoutTemplate:
    """Test cases for LayoutTemplate."""
    
    def test_template_creation(self):
        """Test creating layout template."""
        template = LayoutTemplate(
            template_id="test_template",
            name="Test Template",
            description="A test template",
            layout_type=LayoutType.GRID,
            grid_columns=12,
            widget_slots=[
                {"type": "kpi", "x": 0, "y": 0, "width": 3, "height": 2},
                {"type": "chart", "x": 3, "y": 0, "width": 9, "height": 6}
            ]
        )
        
        assert template.template_id == "test_template"
        assert template.name == "Test Template"
        assert template.layout_type == LayoutType.GRID
        assert len(template.widget_slots) == 2
        assert template.widget_slots[0]["type"] == "kpi"
        assert template.widget_slots[1]["type"] == "chart"


class TestLayoutIntegration:
    """Integration tests for layout management."""
    
    def test_complete_layout_workflow(self):
        """Test complete layout management workflow."""
        manager = AdvancedLayoutManager()
        
        # Apply template
        widget_ids = ["kpi1", "kpi2", "chart1", "chart2"]
        manager.apply_template("executive", widget_ids)
        
        # Add constraints to one widget
        constraints = LayoutConstraints(fixed_position=True)
        manager.widget_constraints["kpi1"] = constraints
        
        # Try to move fixed widget (should fail)
        result = manager.move_widget("kpi1", 6, 0)
        assert result is False
        
        # Move non-fixed widget (should succeed)
        result = manager.move_widget("chart1", 2, 2)
        assert result is True
        
        # Optimize layout
        manager.optimize_layout()
        
        # Export and verify
        exported = manager.export_layout()
        assert len(exported["widget_positions"]) == 4
        assert "kpi1" in exported["widget_constraints"]
    
    def test_responsive_design_workflow(self):
        """Test responsive design workflow."""
        manager = AdvancedLayoutManager()
        
        # Setup desktop layout
        manager.add_widget_position("widget1", 0, 0, 4, 3)
        manager.add_widget_position("widget2", 4, 0, 4, 3)
        manager.add_widget_position("widget3", 8, 0, 4, 3)
        
        # Create mobile layout (stack vertically)
        mobile_layout = ResponsiveLayout(
            breakpoint=ResponsiveBreakpoint.SM,
            grid_columns=6,
            widget_positions={
                "widget1": {"x": 0, "y": 0, "width": 6, "height": 3},
                "widget2": {"x": 0, "y": 3, "width": 6, "height": 3},
                "widget3": {"x": 0, "y": 6, "width": 6, "height": 3}
            }
        )
        
        manager.set_responsive_layout(ResponsiveBreakpoint.SM, mobile_layout)
        
        # Test different breakpoints
        desktop_layout = manager.get_layout_for_breakpoint(ResponsiveBreakpoint.LG)
        mobile_layout_result = manager.get_layout_for_breakpoint(ResponsiveBreakpoint.SM)
        
        # Desktop should have 3 columns
        assert desktop_layout["grid_columns"] == 12
        desktop_positions = desktop_layout["widget_positions"]
        assert desktop_positions["widget1"]["x"] == 0
        assert desktop_positions["widget2"]["x"] == 4
        assert desktop_positions["widget3"]["x"] == 8
        
        # Mobile should stack vertically
        assert mobile_layout_result["grid_columns"] == 6
        mobile_positions = mobile_layout_result["widget_positions"]
        assert mobile_positions["widget1"]["x"] == 0
        assert mobile_positions["widget2"]["x"] == 0
        assert mobile_positions["widget3"]["x"] == 0
        assert mobile_positions["widget2"]["y"] == 3
        assert mobile_positions["widget3"]["y"] == 6


if __name__ == "__main__":
    pytest.main([__file__])