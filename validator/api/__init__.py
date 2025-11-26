"""
REST API for NEDIS synthetic data validation.

This package provides:
- FastAPI-based REST API
- Real-time validation endpoints
- WebSocket support for live updates
- Validation history and management
- Configuration management via API
"""

from .routes import create_app

__all__ = ['create_app']
