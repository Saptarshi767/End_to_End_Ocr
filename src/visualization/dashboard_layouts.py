"""
Advanced dashboard layout management system.

This module provides sophisticated layout management capabilities including
responsive layouts, drag-and-drop positioning, and layout templates.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

from ..core.models import DashboardLayout


logger = logging.getLogger(__name__)


class LayoutType(Enum):
    """Dashboard layout types."""
    GRID = "grid"
    FLEX = "flex"
    MASONRY = "masonry"
    CUSTOM = "custom"


class ResponsiveBreakpoint(Enum):
    """Responsive design breakpoints."""
    XS = "xs"  # < 576px
    SM = "sm"  # >= 576px
    MD = "md"  # >= 768px
    LG = "lg"  # >= 992px
    XL = "xl"  # >= 1200px
    XXL = "xxl"  # >= 1400px


@dataclass
class LayoutConstraints:
    """Layout constraints for widgets."""
    min_width: int = 1
    max_width: int = 12
    min_height: int = 1
    max_height: int = 20
    aspect_ratio: Optional[float] = None
    fixed_position: bool = False


@dataclass
class ResponsiveLayout:
    """Responsive layout configuration for different screen sizes."""
    breakpoint: ResponsiveBreakpoint
    grid_columns: int
    widget_positions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    hidden_widgets: List[str] = field(default_factory=list)


@dataclass
class LayoutTemplate:
    """Predefined layout template."""
    template_id: str
    name: str
    description: str
    layout_type: LayoutType
    grid_columns: int = 12
    widget_slots: List[Dict[str, Any]] = field(default_factory=list)
    responsive_layouts: Dict[str, ResponsiveLayout] = field(default_factory=dict)
    preview_image: Optional[str] = None


class AdvancedLayoutManager:
    """Advanced layout manager with responsive design and templates."""
    
    def __init__(self, grid_columns: int = 12):
        self.grid_columns = grid_columns
        self.layout_type = LayoutType.GRID
        self.widget_positions: Dict[str, Dict[str, int]] = {}
        self.widget_constraints: Dict[str, LayoutConstraints] = {}
        self.responsive_layouts: Dict[str, ResponsiveLayout] = {}
        self.layout_templates: Dict[str, LayoutTemplate] = {}
        self.current_breakpoint = ResponsiveBreakpoint.LG
        
        # Initialize default templates
        self._initialize_default_templates()
    
    def _initialize_default_templates(self) -> None:
        """Initialize default layout templates."""
        
        # Executive Dashboard Template
        executive_template = LayoutTemplate(
            template_id="executive",
            name="Executive Dashboard",
            description="High-level KPIs and summary charts for executives",
            layout_type=LayoutType.GRID,
            grid_columns=12,
            widget_slots=[
                {"type": "kpi", "x": 0, "y": 0, "width": 3, "height": 2},
                {"type": "kpi", "x": 3, "y": 0, "width": 3, "height": 2},
                {"type": "kpi", "x": 6, "y": 0, "width": 3, "height": 2},
                {"type": "kpi", "x": 9, "y": 0, "width": 3, "height": 2},
                {"type": "chart", "x": 0, "y": 2, "width": 8, "height": 6},
                {"type": "chart", "x": 8, "y": 2, "width": 4, "height": 6},
                {"type": "chart", "x": 0, "y": 8, "width": 6, "height": 4},
                {"type": "chart", "x": 6, "y": 8, "width": 6, "height": 4}
            ]
        )
        
        # Analytical Dashboard Template
        analytical_template = LayoutTemplate(
            template_id="analytical",
            name="Analytical Dashboard",
            description="Detailed charts and filters for data analysis",
            layout_type=LayoutType.GRID,
            grid_columns=12,
            widget_slots=[
                {"type": "filter", "x": 0, "y": 0, "width": 12, "height": 1},
                {"type": "chart", "x": 0, "y": 1, "width": 6, "height": 5},
                {"type": "chart", "x": 6, "y": 1, "width": 6, "height": 5},
                {"type": "chart", "x": 0, "y": 6, "width": 4, "height": 4},
                {"type": "chart", "x": 4, "y": 6, "width": 4, "height": 4},
                {"type": "chart", "x": 8, "y": 6, "width": 4, "height": 4},
                {"type": "table", "x": 0, "y": 10, "width": 12, "height": 6}
            ]
        )
        
        # Operational Dashboard Template
        operational_template = LayoutTemplate(
            template_id="operational",
            name="Operational Dashboard",
            description="Real-time monitoring with KPIs and alerts",
            layout_type=LayoutType.GRID,
            grid_columns=12,
            widget_slots=[
                {"type": "kpi", "x": 0, "y": 0, "width": 2, "height": 2},
                {"type": "kpi", "x": 2, "y": 0, "width": 2, "height": 2},
                {"type": "kpi", "x": 4, "y": 0, "width": 2, "height": 2},
                {"type": "kpi", "x": 6, "y": 0, "width": 2, "height": 2},
                {"type": "kpi", "x": 8, "y": 0, "width": 2, "height": 2},
                {"type": "kpi", "x": 10, "y": 0, "width": 2, "height": 2},
                {"type": "chart", "x": 0, "y": 2, "width": 12, "height": 4},
                {"type": "chart", "x": 0, "y": 6, "width": 6, "height": 4},
                {"type": "chart", "x": 6, "y": 6, "width": 6, "height": 4}
            ]
        )
        
        self.layout_templates = {
            "executive": executive_template,
            "analytical": analytical_template,
            "operational": operational_template
        }
    
    def apply_template(self, template_id: str, widget_ids: List[str]) -> bool:
        """Apply layout template to widgets."""
        if template_id not in self.layout_templates:
            logger.error(f"Unknown template: {template_id}")
            return False
        
        template = self.layout_templates[template_id]
        
        # Clear existing positions
        self.widget_positions.clear()
        
        # Apply template positions to widgets
        for i, widget_id in enumerate(widget_ids):
            if i < len(template.widget_slots):
                slot = template.widget_slots[i]
                self.widget_positions[widget_id] = {
                    "x": slot["x"],
                    "y": slot["y"],
                    "width": slot["width"],
                    "height": slot["height"]
                }
        
        self.grid_columns = template.grid_columns
        self.layout_type = template.layout_type
        
        logger.info(f"Applied template {template_id} to {len(widget_ids)} widgets")
        return True
    
    def add_widget_position(self, widget_id: str, x: int, y: int, width: int, height: int,
                           constraints: Optional[LayoutConstraints] = None) -> bool:
        """Add widget position with optional constraints."""
        
        # Apply constraints if provided
        if constraints:
            width = max(constraints.min_width, min(width, constraints.max_width))
            height = max(constraints.min_height, min(height, constraints.max_height))
            
            if constraints.aspect_ratio:
                # Maintain aspect ratio
                calculated_height = int(width / constraints.aspect_ratio)
                if calculated_height != height:
                    height = calculated_height
        
        # Ensure widget fits within grid
        if x + width > self.grid_columns:
            x = max(0, self.grid_columns - width)
        
        # Check for overlaps and resolve
        if self._has_overlap(widget_id, x, y, width, height):
            new_position = self._find_available_position(width, height)
            if new_position:
                x, y = new_position
            else:
                logger.warning(f"Could not find available position for widget {widget_id}")
                return False
        
        self.widget_positions[widget_id] = {
            "x": x,
            "y": y,
            "width": width,
            "height": height
        }
        
        if constraints:
            self.widget_constraints[widget_id] = constraints
        
        return True
    
    def move_widget(self, widget_id: str, new_x: int, new_y: int) -> bool:
        """Move widget to new position."""
        if widget_id not in self.widget_positions:
            return False
        
        current_pos = self.widget_positions[widget_id]
        width = current_pos["width"]
        height = current_pos["height"]
        
        # Check constraints
        if widget_id in self.widget_constraints:
            constraints = self.widget_constraints[widget_id]
            if constraints.fixed_position:
                logger.warning(f"Widget {widget_id} has fixed position")
                return False
        
        # Check if new position is valid
        if new_x + width > self.grid_columns or new_x < 0 or new_y < 0:
            return False
        
        # Check for overlaps
        if self._has_overlap(widget_id, new_x, new_y, width, height):
            # Try to resolve overlap by moving other widgets
            if not self._resolve_overlap(widget_id, new_x, new_y, width, height):
                return False
        
        self.widget_positions[widget_id]["x"] = new_x
        self.widget_positions[widget_id]["y"] = new_y
        
        return True
    
    def resize_widget(self, widget_id: str, new_width: int, new_height: int) -> bool:
        """Resize widget with constraint checking."""
        if widget_id not in self.widget_positions:
            return False
        
        current_pos = self.widget_positions[widget_id]
        x = current_pos["x"]
        y = current_pos["y"]
        
        # Apply constraints
        if widget_id in self.widget_constraints:
            constraints = self.widget_constraints[widget_id]
            new_width = max(constraints.min_width, min(new_width, constraints.max_width))
            new_height = max(constraints.min_height, min(new_height, constraints.max_height))
            
            if constraints.aspect_ratio:
                new_height = int(new_width / constraints.aspect_ratio)
        
        # Check if new size fits
        if x + new_width > self.grid_columns:
            return False
        
        # Check for overlaps
        if self._has_overlap(widget_id, x, y, new_width, new_height):
            return False
        
        self.widget_positions[widget_id]["width"] = new_width
        self.widget_positions[widget_id]["height"] = new_height
        
        return True
    
    def auto_layout_widgets(self, widget_ids: List[str], layout_strategy: str = "balanced") -> None:
        """Automatically layout widgets using specified strategy."""
        
        if layout_strategy == "balanced":
            self._auto_layout_balanced(widget_ids)
        elif layout_strategy == "compact":
            self._auto_layout_compact(widget_ids)
        elif layout_strategy == "grid":
            self._auto_layout_grid(widget_ids)
        else:
            logger.warning(f"Unknown layout strategy: {layout_strategy}")
            self._auto_layout_balanced(widget_ids)
    
    def _auto_layout_balanced(self, widget_ids: List[str]) -> None:
        """Auto-layout with balanced distribution."""
        widgets_per_row = min(3, len(widget_ids))
        widget_width = self.grid_columns // widgets_per_row
        widget_height = 4
        
        for i, widget_id in enumerate(widget_ids):
            row = i // widgets_per_row
            col = i % widgets_per_row
            
            x = col * widget_width
            y = row * widget_height
            
            self.widget_positions[widget_id] = {
                "x": x,
                "y": y,
                "width": widget_width,
                "height": widget_height
            }
    
    def _auto_layout_compact(self, widget_ids: List[str]) -> None:
        """Auto-layout with compact arrangement."""
        current_x = 0
        current_y = 0
        row_height = 0
        
        for widget_id in widget_ids:
            # Default widget size
            width = 4
            height = 3
            
            # Check if widget fits in current row
            if current_x + width > self.grid_columns:
                # Move to next row
                current_x = 0
                current_y += row_height
                row_height = 0
            
            self.widget_positions[widget_id] = {
                "x": current_x,
                "y": current_y,
                "width": width,
                "height": height
            }
            
            current_x += width
            row_height = max(row_height, height)
    
    def _auto_layout_grid(self, widget_ids: List[str]) -> None:
        """Auto-layout in regular grid pattern."""
        cols = 3  # Fixed 3 columns
        widget_width = self.grid_columns // cols
        widget_height = 4
        
        for i, widget_id in enumerate(widget_ids):
            row = i // cols
            col = i % cols
            
            x = col * widget_width
            y = row * widget_height
            
            self.widget_positions[widget_id] = {
                "x": x,
                "y": y,
                "width": widget_width,
                "height": widget_height
            }
    
    def _has_overlap(self, widget_id: str, x: int, y: int, width: int, height: int) -> bool:
        """Check if widget position overlaps with existing widgets."""
        for other_id, other_pos in self.widget_positions.items():
            if other_id == widget_id:
                continue
            
            # Check for overlap
            if (x < other_pos["x"] + other_pos["width"] and
                x + width > other_pos["x"] and
                y < other_pos["y"] + other_pos["height"] and
                y + height > other_pos["y"]):
                return True
        
        return False
    
    def _find_available_position(self, width: int, height: int) -> Optional[Tuple[int, int]]:
        """Find available position for widget of given size."""
        max_y = max([pos["y"] + pos["height"] for pos in self.widget_positions.values()], default=0)
        
        # Try to find position in existing rows first
        for y in range(0, max_y + height):
            for x in range(0, self.grid_columns - width + 1):
                if not self._has_overlap("", x, y, width, height):
                    return (x, y)
        
        # If no position found, add to bottom
        return (0, max_y)
    
    def _resolve_overlap(self, widget_id: str, x: int, y: int, width: int, height: int) -> bool:
        """Try to resolve overlap by moving other widgets."""
        # This is a simplified implementation
        # In practice, you might implement more sophisticated algorithms
        return False
    
    def set_responsive_layout(self, breakpoint: ResponsiveBreakpoint, 
                            layout: ResponsiveLayout) -> None:
        """Set responsive layout for specific breakpoint."""
        self.responsive_layouts[breakpoint.value] = layout
    
    def get_layout_for_breakpoint(self, breakpoint: ResponsiveBreakpoint) -> Dict[str, Any]:
        """Get layout configuration for specific breakpoint."""
        if breakpoint.value in self.responsive_layouts:
            responsive_layout = self.responsive_layouts[breakpoint.value]
            return {
                "grid_columns": responsive_layout.grid_columns,
                "widget_positions": responsive_layout.widget_positions,
                "hidden_widgets": responsive_layout.hidden_widgets
            }
        
        # Return default layout
        return {
            "grid_columns": self.grid_columns,
            "widget_positions": self.widget_positions,
            "hidden_widgets": []
        }
    
    def optimize_layout(self) -> None:
        """Optimize layout by minimizing gaps and improving arrangement."""
        # Sort widgets by y position, then x position
        sorted_widgets = sorted(
            self.widget_positions.items(),
            key=lambda item: (item[1]["y"], item[1]["x"])
        )
        
        # Compact layout by removing gaps
        current_y = 0
        row_widgets = []
        
        for widget_id, pos in sorted_widgets:
            if not row_widgets or pos["y"] == current_y:
                # Same row
                row_widgets.append((widget_id, pos))
            else:
                # New row - process previous row
                self._compact_row(row_widgets, current_y)
                current_y = pos["y"]
                row_widgets = [(widget_id, pos)]
        
        # Process last row
        if row_widgets:
            self._compact_row(row_widgets, current_y)
    
    def _compact_row(self, row_widgets: List[Tuple[str, Dict[str, int]]], y: int) -> None:
        """Compact widgets in a row to remove gaps."""
        # Sort by x position
        row_widgets.sort(key=lambda item: item[1]["x"])
        
        current_x = 0
        for widget_id, pos in row_widgets:
            self.widget_positions[widget_id]["x"] = current_x
            self.widget_positions[widget_id]["y"] = y
            current_x += pos["width"]
    
    def get_layout_config(self) -> DashboardLayout:
        """Get current layout configuration."""
        return DashboardLayout(
            grid_columns=self.grid_columns,
            chart_positions=self.widget_positions.copy(),
            responsive=len(self.responsive_layouts) > 0
        )
    
    def export_layout(self) -> Dict[str, Any]:
        """Export layout configuration for saving/sharing."""
        return {
            "grid_columns": self.grid_columns,
            "layout_type": self.layout_type.value,
            "widget_positions": self.widget_positions,
            "widget_constraints": {
                widget_id: {
                    "min_width": constraints.min_width,
                    "max_width": constraints.max_width,
                    "min_height": constraints.min_height,
                    "max_height": constraints.max_height,
                    "aspect_ratio": constraints.aspect_ratio,
                    "fixed_position": constraints.fixed_position
                }
                for widget_id, constraints in self.widget_constraints.items()
            },
            "responsive_layouts": {
                breakpoint: {
                    "grid_columns": layout.grid_columns,
                    "widget_positions": layout.widget_positions,
                    "hidden_widgets": layout.hidden_widgets
                }
                for breakpoint, layout in self.responsive_layouts.items()
            }
        }
    
    def import_layout(self, layout_config: Dict[str, Any]) -> bool:
        """Import layout configuration."""
        try:
            self.grid_columns = layout_config.get("grid_columns", 12)
            self.layout_type = LayoutType(layout_config.get("layout_type", "grid"))
            self.widget_positions = layout_config.get("widget_positions", {})
            
            # Import constraints
            constraints_data = layout_config.get("widget_constraints", {})
            for widget_id, constraint_dict in constraints_data.items():
                self.widget_constraints[widget_id] = LayoutConstraints(
                    min_width=constraint_dict.get("min_width", 1),
                    max_width=constraint_dict.get("max_width", 12),
                    min_height=constraint_dict.get("min_height", 1),
                    max_height=constraint_dict.get("max_height", 20),
                    aspect_ratio=constraint_dict.get("aspect_ratio"),
                    fixed_position=constraint_dict.get("fixed_position", False)
                )
            
            # Import responsive layouts
            responsive_data = layout_config.get("responsive_layouts", {})
            for breakpoint, layout_dict in responsive_data.items():
                self.responsive_layouts[breakpoint] = ResponsiveLayout(
                    breakpoint=ResponsiveBreakpoint(breakpoint),
                    grid_columns=layout_dict.get("grid_columns", 12),
                    widget_positions=layout_dict.get("widget_positions", {}),
                    hidden_widgets=layout_dict.get("hidden_widgets", [])
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error importing layout: {e}")
            return False
    
    def get_available_templates(self) -> List[Dict[str, str]]:
        """Get list of available layout templates."""
        return [
            {
                "id": template_id,
                "name": template.name,
                "description": template.description,
                "layout_type": template.layout_type.value
            }
            for template_id, template in self.layout_templates.items()
        ]
    
    def add_custom_template(self, template: LayoutTemplate) -> None:
        """Add custom layout template."""
        self.layout_templates[template.template_id] = template
        logger.info(f"Added custom template: {template.template_id}")