"""
Visualization and web interface for NEDIS synthetic data validation.

This package provides:
- Web dashboard for real-time validation monitoring
- Interactive charts and reports
- REST API for validation services
- Real-time WebSocket updates
"""

from .dashboard import ValidationDashboard
from .reports import ReportGenerator

__all__ = ['ValidationDashboard', 'ReportGenerator']
