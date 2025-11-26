#!/usr/bin/env python3
"""
FastAPI routes for NEDIS validation API.

This module provides REST API endpoints for:
- Running validations
- Getting validation results
- Managing validation history
- Real-time WebSocket updates
- Configuration management
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import threading
import time

from ..core.validator import ValidationOrchestrator, validate_async, ValidationResult
from ..core.config import ValidationConfig, get_config
from ..utils.metrics import get_performance_tracker


class ValidationAPI:
    """FastAPI application for validation services"""

    def __init__(self):
        """Initialize the API"""
        self.app = FastAPI(
            title="NEDIS Validation API",
            description="REST API for synthetic data validation",
            version="2.0.0"
        )

        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Active WebSocket connections
        self.active_connections: List[WebSocket] = []

        # Background task for WebSocket updates
        self.websocket_update_task = None

        # Setup routes
        self._setup_routes()

        # Start WebSocket update task
        self._start_websocket_updates()

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "message": "NEDIS Validation API",
                "version": "2.0.0",
                "docs": "/docs",
                "health": "/health"
            }

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "2.0.0"
            }

        @self.app.post("/validate")
        async def run_validation(
            original_db: str,
            synthetic_db: str,
            validation_type: str = "comprehensive",
            sample_size: Optional[int] = None
        ):
            """Run validation"""
            try:
                orchestrator = ValidationOrchestrator()

                if validation_type == "comprehensive":
                    result = await orchestrator.validate_comprehensive(
                        original_db, synthetic_db, sample_size
                    )
                elif validation_type == "statistical":
                    result = orchestrator.validate_statistical(
                        original_db, synthetic_db, sample_size
                    )
                elif validation_type == "patterns":
                    result = orchestrator.validate_patterns(
                        original_db, synthetic_db
                    )
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown validation type: {validation_type}")

                return {
                    "validation_id": result.metadata.get('validation_id', 'unknown'),
                    "status": "completed",
                    "overall_score": result.overall_score,
                    "duration": result.duration,
                    "results": result.results
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/validate/{validation_id}")
        async def get_validation_result(validation_id: str):
            """Get validation result by ID"""
            try:
                orchestrator = ValidationOrchestrator()
                history = orchestrator.get_validation_history(limit=100)

                for result in history:
                    if result.metadata.get('validation_id') == validation_id:
                        return result.to_dict()

                raise HTTPException(status_code=404, detail="Validation not found")

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/validations")
        async def list_validations(limit: int = 10, offset: int = 0):
            """List recent validations"""
            try:
                orchestrator = ValidationOrchestrator()
                history = orchestrator.get_validation_history(limit=limit + offset)

                results = []
                for i, result in enumerate(history[offset:offset + limit]):
                    results.append({
                        "id": result.metadata.get('validation_id', f'validation_{i}'),
                        "type": result.validation_type,
                        "score": result.overall_score,
                        "duration": result.duration,
                        "timestamp": result.start_time.isoformat(),
                        "errors": len(result.errors),
                        "warnings": len(result.warnings)
                    })

                return {
                    "validations": results,
                    "total": len(history),
                    "limit": limit,
                    "offset": offset
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/stats")
        async def get_statistics():
            """Get validation statistics"""
            try:
                orchestrator = ValidationOrchestrator()
                stats = orchestrator.get_validation_stats()

                perf_tracker = get_performance_tracker()
                perf_stats = perf_tracker.get_stats()

                return {
                    "validation_stats": stats,
                    "performance_stats": perf_stats,
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/config")
        async def get_configuration():
            """Get current configuration"""
            try:
                config = get_config()
                return config.to_dict()

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/config")
        async def update_configuration(config_update: Dict[str, Any]):
            """Update configuration (placeholder)"""
            return {"message": "Configuration update not implemented yet", "received": config_update}

        @self.app.websocket("/ws/validation-updates")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            self.active_connections.append(websocket)

            try:
                while True:
                    # Wait for client messages
                    data = await websocket.receive_text()
                    await websocket.send_text(f"Echo: {data}")

            except WebSocketDisconnect:
                self.active_connections.remove(websocket)

    def _start_websocket_updates(self):
        """Start background task for WebSocket updates"""
        def send_periodic_updates():
            while True:
                try:
                    if self.active_connections:
                        # Get current stats
                        orchestrator = ValidationOrchestrator()
                        stats = orchestrator.get_validation_stats()

                        update_data = {
                            "type": "stats_update",
                            "timestamp": datetime.now().isoformat(),
                            "data": stats
                        }

                        # Send to all connected clients
                        disconnected_connections = []
                        for connection in self.active_connections:
                            try:
                                asyncio.run(connection.send_text(json.dumps(update_data)))
                            except:
                                disconnected_connections.append(connection)

                        # Remove disconnected connections
                        for connection in disconnected_connections:
                            self.active_connections.remove(connection)

                    time.sleep(5)  # Update every 5 seconds

                except Exception as e:
                    print(f"WebSocket update error: {e}")
                    time.sleep(10)  # Wait longer on error

        self.websocket_update_task = threading.Thread(target=send_periodic_updates, daemon=True)
        self.websocket_update_task.start()

    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
        """Run the API server"""
        import uvicorn

        print("🚀 Starting NEDIS Validation API..."        print(f"📍 API will be available at: http://{host}:{port}")
        print(f"📚 API documentation: http://{host}:{port}/docs")
        print(f"🔄 ReDoc: http://{host}:{port}/redoc")

        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info" if not debug else "debug"
        )


def create_app() -> ValidationAPI:
    """Create and return a validation API instance"""
    return ValidationAPI()


if __name__ == "__main__":
    api = create_app()
    api.run(debug=True)
