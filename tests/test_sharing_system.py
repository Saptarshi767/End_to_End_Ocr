"""
Tests for dashboard sharing and collaboration system
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.sharing.share_manager import ShareManager, ShareType, Permission, ShareLink
from src.sharing.embed_service import EmbedService, WidgetType, EmbedTheme
from src.core.models import Dashboard, Chart, KPI
from src.core.exceptions import SharingError


class TestShareManager:
    """Test cases for ShareManager"""
    
    @pytest.fixture
    def share_manager(self):
        """Create ShareManager instance"""
        return ShareManager("https://test.example.com")
    
    @pytest.fixture
    def sample_dashboard_id(self):
        """Sample dashboard ID"""
        return "dashboard_123"
    
    @pytest.fixture
    def sample_owner_id(self):
        """Sample owner ID"""
        return "user_123"
    
    def test_create_share_link(self, share_manager, sample_dashboard_id, sample_owner_id):
        """Test creating a share link"""
        
        share_link = share_manager.create_share_link(
            dashboard_id=sample_dashboard_id,
            owner_id=sample_owner_id,
            share_type=ShareType.SECURE_LINK,
            permissions=[Permission.VIEW, Permission.INTERACT],
            expires_in_hours=24,
            password="test123"
        )
        
        assert isinstance(share_link, ShareLink)
        assert share_link.dashboard_id == sample_dashboard_id
        assert share_link.owner_id == sample_owner_id
        assert share_link.share_type == ShareType.SECURE_LINK
        assert Permission.VIEW in share_link.permissions
        assert Permission.INTERACT in share_link.permissions
        assert share_link.password_hash is not None
        assert share_link.expires_at is not None
        assert "https://test.example.com/shared/" in share_link.share_url
    
    def test_create_public_link(self, share_manager, sample_dashboard_id, sample_owner_id):
        """Test creating a public share link"""
        
        share_link = share_manager.create_share_link(
            dashboard_id=sample_dashboard_id,
            owner_id=sample_owner_id,
            share_type=ShareType.PUBLIC_LINK,
            permissions=[Permission.VIEW]
        )
        
        assert share_link.share_type == ShareType.PUBLIC_LINK
        assert share_link.password_hash is None
        assert share_link.expires_at is None
    
    def test_validate_access_success(self, share_manager, sample_dashboard_id, sample_owner_id):
        """Test successful access validation"""
        
        share_link = share_manager.create_share_link(
            dashboard_id=sample_dashboard_id,
            owner_id=sample_owner_id,
            password="test123"
        )
        
        # Valid access
        assert share_manager.validate_access(
            share_link.share_id,
            password="test123",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        ) == True
        
        # Check access count increased
        updated_link = share_manager.get_share_link(share_link.share_id)
        assert updated_link.access_count == 1
        assert updated_link.last_accessed is not None
    
    def test_validate_access_wrong_password(self, share_manager, sample_dashboard_id, sample_owner_id):
        """Test access validation with wrong password"""
        
        share_link = share_manager.create_share_link(
            dashboard_id=sample_dashboard_id,
            owner_id=sample_owner_id,
            password="test123"
        )
        
        # Wrong password
        assert share_manager.validate_access(
            share_link.share_id,
            password="wrong"
        ) == False
        
        # No password provided
        assert share_manager.validate_access(share_link.share_id) == False
    
    def test_validate_access_expired(self, share_manager, sample_dashboard_id, sample_owner_id):
        """Test access validation for expired link"""
        
        share_link = share_manager.create_share_link(
            dashboard_id=sample_dashboard_id,
            owner_id=sample_owner_id,
            expires_in_hours=-1  # Already expired
        )
        
        assert share_manager.validate_access(share_link.share_id) == False
    
    def test_validate_access_max_reached(self, share_manager, sample_dashboard_id, sample_owner_id):
        """Test access validation when max access reached"""
        
        share_link = share_manager.create_share_link(
            dashboard_id=sample_dashboard_id,
            owner_id=sample_owner_id,
            max_access=1
        )
        
        # First access should succeed
        assert share_manager.validate_access(share_link.share_id) == True
        
        # Second access should fail
        assert share_manager.validate_access(share_link.share_id) == False
    
    def test_revoke_share_link(self, share_manager, sample_dashboard_id, sample_owner_id):
        """Test revoking a share link"""
        
        share_link = share_manager.create_share_link(
            dashboard_id=sample_dashboard_id,
            owner_id=sample_owner_id
        )
        
        # Revoke by owner
        assert share_manager.revoke_share_link(share_link.share_id, sample_owner_id) == True
        
        # Link should be inactive
        updated_link = share_manager.get_share_link(share_link.share_id)
        assert updated_link.is_active == False
        
        # Access should fail
        assert share_manager.validate_access(share_link.share_id) == False
    
    def test_revoke_unauthorized(self, share_manager, sample_dashboard_id, sample_owner_id):
        """Test unauthorized revoke attempt"""
        
        share_link = share_manager.create_share_link(
            dashboard_id=sample_dashboard_id,
            owner_id=sample_owner_id
        )
        
        # Try to revoke with different user
        assert share_manager.revoke_share_link(share_link.share_id, "other_user") == False
        
        # Link should still be active
        updated_link = share_manager.get_share_link(share_link.share_id)
        assert updated_link.is_active == True
    
    def test_list_dashboard_shares(self, share_manager, sample_dashboard_id, sample_owner_id):
        """Test listing shares for a dashboard"""
        
        # Create multiple shares
        share1 = share_manager.create_share_link(sample_dashboard_id, sample_owner_id)
        share2 = share_manager.create_share_link(sample_dashboard_id, sample_owner_id)
        share3 = share_manager.create_share_link("other_dashboard", sample_owner_id)
        
        shares = share_manager.list_dashboard_shares(sample_dashboard_id, sample_owner_id)
        
        assert len(shares) == 2
        share_ids = [s.share_id for s in shares]
        assert share1.share_id in share_ids
        assert share2.share_id in share_ids
        assert share3.share_id not in share_ids
    
    def test_get_share_analytics(self, share_manager, sample_dashboard_id, sample_owner_id):
        """Test getting share analytics"""
        
        share_link = share_manager.create_share_link(
            dashboard_id=sample_dashboard_id,
            owner_id=sample_owner_id
        )
        
        # Generate some access
        share_manager.validate_access(share_link.share_id, ip_address="192.168.1.1")
        share_manager.validate_access(share_link.share_id, ip_address="192.168.1.2")
        
        analytics = share_manager.get_share_analytics(share_link.share_id, sample_owner_id)
        
        assert analytics["share_id"] == share_link.share_id
        assert analytics["total_accesses"] == 2
        assert analytics["unique_ips"] == 2
        assert "recent_accesses" in analytics
        assert len(analytics["recent_accesses"]) == 2
    
    def test_update_share_settings(self, share_manager, sample_dashboard_id, sample_owner_id):
        """Test updating share settings"""
        
        share_link = share_manager.create_share_link(
            dashboard_id=sample_dashboard_id,
            owner_id=sample_owner_id
        )
        
        # Update settings
        success = share_manager.update_share_settings(
            share_link.share_id,
            sample_owner_id,
            permissions=[Permission.VIEW, Permission.EXPORT],
            expires_in_hours=48,
            password="newpassword"
        )
        
        assert success == True
        
        updated_link = share_manager.get_share_link(share_link.share_id)
        assert Permission.EXPORT in updated_link.permissions
        assert updated_link.password_hash is not None
        assert updated_link.expires_at is not None
    
    def test_cleanup_expired_shares(self, share_manager, sample_dashboard_id, sample_owner_id):
        """Test cleanup of expired shares"""
        
        # Create expired share
        expired_share = share_manager.create_share_link(
            dashboard_id=sample_dashboard_id,
            owner_id=sample_owner_id,
            expires_in_hours=-1
        )
        
        # Create active share
        active_share = share_manager.create_share_link(
            dashboard_id=sample_dashboard_id,
            owner_id=sample_owner_id
        )
        
        # Cleanup
        cleaned_count = share_manager.cleanup_expired_shares()
        
        assert cleaned_count == 1
        assert share_manager.get_share_link(expired_share.share_id) is None
        assert share_manager.get_share_link(active_share.share_id) is not None


class TestEmbedService:
    """Test cases for EmbedService"""
    
    @pytest.fixture
    def embed_service(self):
        """Create EmbedService instance"""
        return EmbedService("https://test.example.com")
    
    @pytest.fixture
    def sample_dashboard(self):
        """Create sample dashboard"""
        charts = [
            Chart(
                id="chart1",
                title="Test Chart",
                chart_type="bar",
                data={"labels": ["A", "B"], "values": [1, 2]},
                options={}
            )
        ]
        
        kpis = [
            KPI(
                id="kpi1",
                name="Test KPI",
                value=100,
                unit="units",
                trend="up",
                change=5.0
            )
        ]
        
        return Dashboard(
            id="dash1",
            charts=charts,
            kpis=kpis,
            filters=[],
            layout=None,
            export_options=[]
        )
    
    def test_create_chart_embed(self, embed_service):
        """Test creating chart embed widget"""
        
        widget = embed_service.create_chart_embed(
            dashboard_id="dash1",
            chart_id="chart1",
            title="Embedded Chart",
            theme=EmbedTheme.DARK,
            width=800,
            height=600
        )
        
        assert widget.widget_type == WidgetType.CHART
        assert widget.dashboard_id == "dash1"
        assert widget.source_id == "chart1"
        assert widget.title == "Embedded Chart"
        assert widget.theme == EmbedTheme.DARK
        assert widget.width == 800
        assert widget.height == 600
    
    def test_create_kpi_embed(self, embed_service):
        """Test creating KPI embed widget"""
        
        widget = embed_service.create_kpi_embed(
            dashboard_id="dash1",
            kpi_id="kpi1",
            customization={"auto_refresh": True, "refresh_interval": 60}
        )
        
        assert widget.widget_type == WidgetType.KPI
        assert widget.auto_refresh == True
        assert widget.refresh_interval == 60
    
    def test_create_dashboard_embed(self, embed_service):
        """Test creating full dashboard embed widget"""
        
        widget = embed_service.create_dashboard_embed(
            dashboard_id="dash1",
            title="Full Dashboard",
            width=1200,
            height=800
        )
        
        assert widget.widget_type == WidgetType.DASHBOARD
        assert widget.source_id == "dash1"
        assert widget.title == "Full Dashboard"
        assert widget.width == 1200
        assert widget.height == 800
    
    def test_generate_embed_code(self, embed_service):
        """Test generating HTML embed code"""
        
        widget = embed_service.create_chart_embed(
            dashboard_id="dash1",
            chart_id="chart1"
        )
        
        embed_code = embed_service.generate_embed_code(widget.widget_id)
        
        assert isinstance(embed_code, str)
        assert f'data-widget-id="{widget.widget_id}"' in embed_code
        assert 'iframe' in embed_code
        assert 'https://test.example.com/embed/' in embed_code
        assert 'window.addEventListener' in embed_code  # Auto-resize script
    
    def test_generate_embed_code_responsive(self, embed_service):
        """Test generating responsive embed code"""
        
        widget = embed_service.create_chart_embed(
            dashboard_id="dash1",
            chart_id="chart1"
        )
        
        embed_code = embed_service.generate_embed_code(
            widget.widget_id,
            responsive=True,
            lazy_load=True
        )
        
        assert 'width="100%"' in embed_code
        assert 'loading="lazy"' in embed_code
        assert 'max-width: 100%' in embed_code
    
    def test_generate_javascript_embed(self, embed_service):
        """Test generating JavaScript embed code"""
        
        widget = embed_service.create_kpi_embed(
            dashboard_id="dash1",
            kpi_id="kpi1"
        )
        
        js_code = embed_service.generate_javascript_embed(
            widget.widget_id,
            container_id="my-widget"
        )
        
        assert isinstance(js_code, str)
        assert 'id="my-widget"' in js_code
        assert f'widgetId: \'{widget.widget_id}\'' in js_code
        assert 'fetch(' in js_code
        assert 'loadWidget()' in js_code
    
    @patch('src.core.services.DashboardService')
    def test_get_widget_data_chart(self, mock_dashboard_service, embed_service, sample_dashboard):
        """Test getting chart widget data"""
        
        # Mock dashboard service
        mock_dashboard_service.return_value.get_dashboard.return_value = sample_dashboard
        
        widget = embed_service.create_chart_embed(
            dashboard_id="dash1",
            chart_id="chart1"
        )
        
        widget_data = embed_service.get_widget_data(widget.widget_id)
        
        assert widget_data["type"] == "chart"
        assert widget_data["chart_type"] == "bar"
        assert "data" in widget_data
        assert widget_data["theme"] == "light"
    
    @patch('src.core.services.DashboardService')
    def test_get_widget_data_kpi(self, mock_dashboard_service, embed_service, sample_dashboard):
        """Test getting KPI widget data"""
        
        # Mock dashboard service
        mock_dashboard_service.return_value.get_dashboard.return_value = sample_dashboard
        
        widget = embed_service.create_kpi_embed(
            dashboard_id="dash1",
            kpi_id="kpi1"
        )
        
        widget_data = embed_service.get_widget_data(widget.widget_id)
        
        assert widget_data["type"] == "kpi"
        assert widget_data["value"] == 100
        assert widget_data["unit"] == "units"
        assert widget_data["trend"] == "up"
    
    def test_get_widget_data_nonexistent(self, embed_service):
        """Test getting data for nonexistent widget"""
        
        with pytest.raises(ValueError) as exc_info:
            embed_service.get_widget_data("nonexistent")
        
        assert "not found" in str(exc_info.value)
    
    def test_update_widget(self, embed_service):
        """Test updating widget configuration"""
        
        widget = embed_service.create_chart_embed(
            dashboard_id="dash1",
            chart_id="chart1"
        )
        
        success = embed_service.update_widget(
            widget.widget_id,
            title="Updated Title",
            theme=EmbedTheme.DARK,
            width=1000,
            customization={"show_title": False}
        )
        
        assert success == True
        
        updated_widget = embed_service.embed_widgets[widget.widget_id]
        assert updated_widget.title == "Updated Title"
        assert updated_widget.theme == EmbedTheme.DARK
        assert updated_widget.width == 1000
        assert updated_widget.show_title == False
    
    def test_delete_widget(self, embed_service):
        """Test deleting embed widget"""
        
        widget = embed_service.create_chart_embed(
            dashboard_id="dash1",
            chart_id="chart1"
        )
        
        assert embed_service.delete_widget(widget.widget_id) == True
        assert widget.widget_id not in embed_service.embed_widgets
        
        # Try to delete again
        assert embed_service.delete_widget(widget.widget_id) == False
    
    def test_list_dashboard_widgets(self, embed_service):
        """Test listing widgets for a dashboard"""
        
        widget1 = embed_service.create_chart_embed("dash1", "chart1")
        widget2 = embed_service.create_kpi_embed("dash1", "kpi1")
        widget3 = embed_service.create_chart_embed("dash2", "chart2")
        
        widgets = embed_service.list_dashboard_widgets("dash1")
        
        assert len(widgets) == 2
        widget_ids = [w.widget_id for w in widgets]
        assert widget1.widget_id in widget_ids
        assert widget2.widget_id in widget_ids
        assert widget3.widget_id not in widget_ids
    
    def test_customization_options(self, embed_service):
        """Test applying customization options"""
        
        customization = {
            "auto_refresh": True,
            "refresh_interval": 120,
            "show_title": False,
            "show_legend": True,
            "custom_css": ".widget { border: 1px solid red; }",
            "allowed_domains": ["example.com", "test.com"]
        }
        
        widget = embed_service.create_chart_embed(
            dashboard_id="dash1",
            chart_id="chart1",
            customization=customization
        )
        
        assert widget.auto_refresh == True
        assert widget.refresh_interval == 120
        assert widget.show_title == False
        assert widget.show_legend == True
        assert widget.custom_css == ".widget { border: 1px solid red; }"
        assert widget.allowed_domains == ["example.com", "test.com"]