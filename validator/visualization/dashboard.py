#!/usr/bin/env python3
"""
Web Dashboard for NEDIS Synthetic Data Validation.

This module provides a modern web interface for:
- Real-time validation monitoring
- Interactive charts and visualizations
- Validation history and results
- Configuration management
- Performance metrics display
"""

import asyncio
import logging
import math

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask
import pandas as pd
import numpy as np
import threading
import time
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime, timezone

# Import will be done at runtime to avoid circular imports
from ..core.config import get_config
from ..utils.metrics import get_performance_tracker


class ValidationDashboard:
    """Web dashboard for validation system"""

    def __init__(self, host: str = '0.0.0.0', port: int = 8050):
        """
        Initialize the dashboard

        Args:
            host: Host to bind the server to
            port: Port to bind the server to
        """
        self.host = host
        self.port = port

        # Console logging setup for debugging progress updates
        self.logger = logging.getLogger('validator.visualization.dashboard')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[Dashboard] %(asctime)s %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # Track last messages to avoid noisy duplicate logs
        self._last_progress_log: Optional[str] = None
        self._last_metrics_snapshot: Optional[Dict[str, Any]] = None
        self._state_lock = threading.Lock()
        self._pending_state_update: Optional[Dict[str, Any]] = None
        self._current_job: Optional[threading.Thread] = None
        self._active_validation_id: Optional[str] = None

        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            title='NEDIS Validation Dashboard',
            suppress_callback_exceptions=True
        )

        # Import at runtime to avoid circular imports
        from ..core.validator import ValidationOrchestrator, get_orchestrator
        self.orchestrator = get_orchestrator()
        self.config = get_config()
        self.performance_tracker = get_performance_tracker()

        # Setup layout and callbacks
        self._setup_layout()
        self._setup_callbacks()

        # Start background update thread
        self._start_background_updates()

    def _setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("🩺 NEDIS Synthetic Data Validation Dashboard",
                           className="text-center mb-4 mt-4")
                ])
            ]),

            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🎯 Validation Controls"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("🚀 Start Validation",
                                             id="start-validation-btn",
                                             color="primary",
                                             className="me-2"),
                                    dbc.Button("📊 View Results",
                                             id="view-results-btn",
                                             color="secondary",
                                             className="me-2"),
                                    dbc.Button("⚙️ Settings",
                                             id="settings-btn",
                                             color="info")
                                ])
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Original DB:", className="mt-3"),
                                    dcc.Input(
                                        id="original-db-input",
                                        type="text",
                                        value="nedis_data.duckdb",
                                        className="form-control"
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Synthetic DB:", className="mt-3"),
                                    dcc.Input(
                                        id="synthetic-db-input",
                                        type="text",
                                        value="nedis_synth_2017.duckdb",
                                        className="form-control",
                                        placeholder="Real synthetic database"
                                    )
                                ], width=6)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Sample Size:", className="mt-3"),
                                    dcc.Slider(
                                        id="sample-size-slider",
                                        min=1000,
                                        max=100000,
                                        step=1000,
                                        value=50000,
                                        marks={i: f"{i:,}" for i in range(0, 100001, 20000)}
                                    )
                                ])
                            ])
                        ])
                    ])
                ])
            ], className="mb-4"),

            # Live Validation Status
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📈 Live Validation Status"),
                        dbc.CardBody([
                            html.Div(id="validation-status", children="Ready to validate"),
                            html.Div(id="current-progress", className="mt-3"),
                            dbc.Progress(id="validation-progress", value=0, className="mt-2")
                        ])
                    ])
                ])
            ], className="mb-4"),

            # Metrics Cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📊 Validation Score", className="card-title"),
                            html.H2(id="overall-score", children="N/A", className="text-primary"),
                            html.P("Overall validation score", className="card-text")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("⏱️ Duration", className="card-title"),
                            html.H2(id="validation-duration", children="N/A", className="text-success"),
                            html.P("Last validation time", className="card-text")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🔍 Validations", className="card-title"),
                            html.H2(id="total-validations", children="0", className="text-info"),
                            html.P("Total validations run", className="card-text")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📈 Performance", className="card-title"),
                            html.H2(id="avg-query-time", children="N/A", className="text-warning"),
                            html.P("Avg query time (s)", className="card-text")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),

            # Charts Row 1
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Validation Scores Over Time"),
                        dbc.CardBody([
                            dcc.Graph(id="score-timeline-chart")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📈 Performance Metrics"),
                        dbc.CardBody([
                            dcc.Graph(id="performance-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),

            # Charts Row 2
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🔍 Validation Categories"),
                        dbc.CardBody([
                            dcc.Graph(id="category-breakdown-chart")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📋 Recent Validation History"),
                        dbc.CardBody([
                            html.Div(id="validation-history-table")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),

            # Database Comparison
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            dbc.Row([
                                dbc.Col([
                                    html.H5("🔬 Database Column Comparison", className="mb-0")
                                ], width=8),
                                dbc.Col([
                                    dbc.Button("🔄 Compare Tables",
                                             id="compare-tables-btn",
                                             color="outline-primary",
                                             size="sm",
                                             className="float-end")
                                ], width=4)
                            ])
                        ]),
                        dbc.CardBody([
                            # Help text
                            dbc.Alert([
                                html.H6("🔬 Real Data Comparison", className="alert-heading"),
                                html.P([
                                    "Comparing REAL data: ",
                                    html.Code("nedis_data.duckdb"), " (9.1M records) vs ",
                                    html.Code("nedis_synth_2017.duckdb"), " (100K synthetic records). ",
                                    "Available table: ",
                                    html.Strong("nedis2017"), " with 87 columns including vital signs, demographics, and clinical data. ",
                                    "Click '🔄 Compare Tables' to load and analyze!"
                                ], className="mb-0")
                            ], color="success", className="mb-3"),

                            # Table Selection Controls
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Select Tables to Compare:", className="fw-bold mb-2"),
                                    dcc.Dropdown(
                                        id="table-selection-dropdown",
                                        multi=True,
                                        placeholder="Select tables to compare...",
                                        className="mb-3"
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Comparison Type:", className="fw-bold mb-2"),
                                    dcc.RadioItems(
                                        id="comparison-type-radio",
                                        options=[
                                            {'label': 'All Columns', 'value': 'all'},
                                            {'label': 'Numeric Only', 'value': 'numeric'},
                                            {'label': 'Categorical Only', 'value': 'categorical'}
                                        ],
                                        value='all',
                                        className="mb-3"
                                    )
                                ], width=6)
                            ]),

                            # Comparison Results - Fixed container to prevent disappearing on scroll
                            dbc.Card([
                                dbc.CardBody([
                                    html.Div(id="statistical-differences-table", className="table-responsive",
                                           style={'minHeight': '200px'})
                                ])
                            ], className="mt-3", style={'position': 'relative', 'zIndex': 1}),

                            # Export Options
                            dbc.Row([
                                dbc.Col([
                                    dbc.ButtonGroup([
                                        dbc.Button("📊 Export to CSV",
                                                 id="export-csv-btn",
                                                 color="outline-success",
                                                 size="sm"),
                                        dbc.Button("📈 Show Charts",
                                                 id="show-charts-btn",
                                                 color="outline-info",
                                                 size="sm"),
                                        dbc.Button("📄 Export Charts to PDF",
                                                 id="export-pdf-btn",
                                                 color="outline-primary",
                                                 size="sm")
                                    ])
                                ], className="text-end mt-3")
                            ])
                        ])
                    ])
                ])
            ], className="mb-4"),

            # Settings Modal
            dbc.Modal([
                dbc.ModalHeader("⚙️ Dashboard Settings"),
                dbc.ModalBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Auto-refresh interval (seconds):"),
                            dcc.Slider(
                                id="refresh-interval-slider",
                                min=1,
                                max=30,
                                step=1,
                                value=5,
                                marks={i: str(i) for i in range(1, 31, 5)}
                            )
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Chart theme:"),
                            dcc.Dropdown(
                                id="chart-theme-dropdown",
                                options=[
                                    {'label': 'Default', 'value': 'default'},
                                    {'label': 'Dark', 'value': 'dark'},
                                    {'label': 'Light', 'value': 'light'}
                                ],
                                value='default'
                            )
                        ])
                    ])
                ]),
                dbc.ModalFooter([
                    dbc.Button("Save Settings", id="save-settings-btn", color="primary"),
                    dbc.Button("Close", id="close-settings-btn", color="secondary")
                ])
            ], id="settings-modal", size="lg"),

            # Charts Modal for Column Comparison
            dbc.Modal([
                dbc.ModalHeader("📈 Column Distribution Comparison Charts"),
                dbc.ModalBody([
                    dcc.Loading([
                        html.Div(id="comparison-charts-content")
                    ])
                ]),
                dbc.ModalFooter([
                    dbc.Button("Close", id="close-charts-btn", color="secondary")
                ])
            ], id="charts-modal", size="xl"),

            # Interval for live updates
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # 5 seconds
                n_intervals=0
            ),

            # Store for validation state
            dcc.Store(id='validation-state', data={}),

            # Store for charts data (for PDF export)
            dcc.Store(id='charts-data', data={}),

            # Download components
            dcc.Download(id="download-pdf"),
            dcc.Download(id="download-csv")

        ], fluid=True)

    def _setup_callbacks(self):
        """Setup dashboard callbacks"""

        @self.app.callback(
            Output('validation-status', 'children'),
            Output('current-progress', 'children'),
            Output('validation-progress', 'value'),
            Output('validation-state', 'data'),
            Input('start-validation-btn', 'n_clicks'),
            Input('interval-component', 'n_intervals'),
            State('original-db-input', 'value'),
            State('synthetic-db-input', 'value'),
            State('sample-size-slider', 'value'),
            State('validation-state', 'data'),
            prevent_initial_call=True
        )
        def update_validation_status(start_clicks, n_intervals, original_db, synthetic_db, sample_size, current_state):
            """Update validation status and progress"""
            current_state = current_state or {}
            pending_update = self._consume_pending_state_update()
            if pending_update:
                updated_state = {**current_state, **pending_update,
                                 'last_update': datetime.now(timezone.utc).isoformat()}
                return self._state_to_outputs(updated_state)

            ctx = dash.callback_context

            if not ctx.triggered:
                return self._state_to_outputs(current_state)

            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if trigger_id == 'start-validation-btn':
                # Start new validation
                formatted_samples = f"{sample_size:,}" if isinstance(sample_size, int) else 'default'
                self._log_progress(
                    f"Validation requested via dashboard: original='{original_db}', synthetic='{synthetic_db}', samples={formatted_samples}"
                )

                validation_id = self._start_validation_job(original_db, synthetic_db, sample_size)

                if validation_id is None:
                    updated_state = {
                        **current_state,
                        'status': 'running',
                        'message': current_state.get('message') or 'Validation already in progress...'
                    }
                    return self._state_to_outputs(updated_state)

                updated_state = {
                    'status': 'running',
                    'original_db': original_db,
                    'synthetic_db': synthetic_db,
                    'sample_size': sample_size,
                    'validation_id': validation_id,
                    'progress': 10,
                    'message': 'Starting validation...',
                    'started_at': datetime.now(timezone.utc).isoformat(),
                    'last_update': datetime.now(timezone.utc).isoformat()
                }

                return self._state_to_outputs(updated_state)

            if trigger_id == 'interval-component':
                pending_update = self._consume_pending_state_update()
                if pending_update:
                    updated_state = {**current_state, **pending_update,
                                     'last_update': datetime.now(timezone.utc).isoformat()}
                    return self._state_to_outputs(updated_state)

                if current_state.get('status') == 'running':
                    previous_progress = current_state.get('progress', 10)
                    progress = min(previous_progress + 5, 90)
                    samples_value = current_state.get('sample_size', sample_size)
                    formatted_samples = f"{samples_value:,}" if isinstance(samples_value, int) else 'default'
                    message = f"Running validation with {formatted_samples} samples..."

                    self._log_progress(
                        f"Validation still running (interval={n_intervals}): original='{current_state.get('original_db', original_db)}', "
                        f"synthetic='{current_state.get('synthetic_db', synthetic_db)}', samples={formatted_samples}"
                    )

                    updated_state = {**current_state,
                                     'progress': progress,
                                     'message': message,
                                     'last_update': datetime.now(timezone.utc).isoformat()}
                    return self._state_to_outputs(updated_state)

                return self._state_to_outputs(current_state)

            return self._state_to_outputs(current_state)

        @self.app.callback(
            Output('overall-score', 'children'),
            Output('validation-duration', 'children'),
            Output('total-validations', 'children'),
            Output('avg-query-time', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_metrics(n_intervals):
            """Update dashboard metrics"""
            try:
                # Get orchestrator stats
                stats = self.orchestrator.get_validation_stats()
                total_validations = stats.get('total_validations', 0)

                # Get performance stats
                perf_stats = self.performance_tracker.get_stats()
                avg_query_time = perf_stats.get('query_avg_duration', 0)

                metrics_payload = {
                    'avg_score': round(stats.get('average_score', 0), 2),
                    'query_count': perf_stats.get('query_count', 0),
                    'total_validations': total_validations,
                    'avg_query_time': round(avg_query_time, 3)
                }
                self._log_metrics(metrics_payload)

                return (
                    f"{metrics_payload['avg_score']:.1f}",
                    f"{metrics_payload['query_count']}",
                    f"{metrics_payload['total_validations']}",
                    f"{metrics_payload['avg_query_time']:.3f}"
                )
            except Exception as e:
                self.logger.exception("Failed to update dashboard metrics")
                return "N/A", "N/A", "0", "N/A"

        @self.app.callback(
            Output('score-timeline-chart', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_score_timeline(n_intervals):
            """Update score timeline chart"""
            try:
                # Get validation history
                history = self.orchestrator.get_validation_history(limit=10)

                if not history:
                    return go.Figure()

                # Create timeline data
                timestamps = [result.start_time for result in history]
                scores = [result.overall_score for result in history]
                types = [result.validation_type for result in history]

                fig = px.line(
                    x=timestamps,
                    y=scores,
                    color=types,
                    title="Validation Scores Over Time",
                    labels={'x': 'Time', 'y': 'Score'}
                )

                return fig
            except Exception as e:
                self.logger.exception("Failed to update score timeline chart")
                return go.Figure()

        @self.app.callback(
            Output('performance-chart', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_performance_chart(n_intervals):
            """Update performance chart"""
            try:
                perf_stats = self.performance_tracker.get_stats()

                metrics = [
                    'Query Count',
                    'Average Query Duration',
                    'Memory Usage (MB)',
                    'CPU Usage (%)'
                ]

                values = [
                    perf_stats.get('query_count', 0),
                    perf_stats.get('query_avg_duration', 0),
                    perf_stats.get('memory_avg_mb', 0),
                    perf_stats.get('cpu_avg_percent', 0)
                ]

                fig = go.Figure(data=[
                    go.Bar(
                        x=metrics,
                        y=values,
                        marker_color=['blue', 'green', 'orange', 'red']
                    )
                ])

                fig.update_layout(
                    title="System Performance Metrics",
                    xaxis_title="Metric",
                    yaxis_title="Value"
                )

                return fig
            except Exception as e:
                self.logger.exception("Failed to update performance chart")
                return go.Figure()

        @self.app.callback(
            Output('category-breakdown-chart', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_category_breakdown(n_intervals):
            """Update category breakdown chart"""
            try:
                stats = self.orchestrator.get_validation_stats()
                validation_types = stats.get('validation_types', {})

                if not validation_types:
                    return go.Figure()

                fig = px.pie(
                    values=list(validation_types.values()),
                    names=list(validation_types.keys()),
                    title="Validation Types Distribution"
                )

                return fig
            except Exception as e:
                self.logger.exception("Failed to update category breakdown chart")
                return go.Figure()

        @self.app.callback(
            Output('validation-history-table', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_validation_history(n_intervals):
            """Update validation history table"""
            try:
                history = self.orchestrator.get_validation_history(limit=5)

                if not history:
                    return html.P("No validation history available")

                table_rows = []
                for i, result in enumerate(history, 1):
                    table_rows.append(
                        dbc.Row([
                            dbc.Col(f"{result.validation_type}", width=3),
                            dbc.Col(f"{result.overall_score:.1f}", width=2),
                            dbc.Col(f"{result.duration:.1f}s", width=2),
                            dbc.Col(f"{result.start_time.strftime('%H:%M:%S')}", width=3),
                            dbc.Col(f"{len(result.errors)} errors", width=2)
                        ], className="mb-2")
                    )

                return table_rows
            except Exception as e:
                self.logger.exception("Failed to update validation history table")
                return html.P(f"Error loading history: {str(e)}")

        @self.app.callback(
            Output('statistical-differences-table', 'children'),
            Input('validation-state', 'data'),
            Input('compare-tables-btn', 'n_clicks'),
            Input('table-selection-dropdown', 'value'),
            Input('comparison-type-radio', 'value'),
            State('original-db-input', 'value'),
            State('synthetic-db-input', 'value'),
            prevent_initial_call=True
        )
        def update_statistical_differences(state, n_clicks, selected_tables, comparison_type, original_db, synthetic_db):
            """Unified callback for database comparison table - handles both validation state and immediate comparison"""
            ctx = dash.callback_context

            if not ctx.triggered:
                return html.P("No data to compare. Either run a validation or click 'Compare Tables' button.")

            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

            # Handle immediate comparison (from Compare Tables button)
            if trigger_id in ['compare-tables-btn', 'table-selection-dropdown', 'comparison-type-radio']:
                if not n_clicks or not selected_tables:
                    return html.P("Select tables to compare and click 'Compare Tables' button.")

                return self._generate_immediate_comparison(selected_tables, comparison_type, original_db, synthetic_db)

            # Handle validation state (existing functionality)
            elif trigger_id == 'validation-state' and state:
                original_db = state.get('original_db')
                synthetic_db = state.get('synthetic_db')

                if not original_db or not synthetic_db:
                    return html.P("Database paths not found in validation state.")

                return self._generate_validation_comparison(original_db, synthetic_db)

            return html.P("No data to compare. Either run a validation or click 'Compare Tables' button.")

        @self.app.callback(
            Output('table-selection-dropdown', 'options'),
            Input('compare-tables-btn', 'n_clicks'),
            State('original-db-input', 'value'),
            State('synthetic-db-input', 'value'),
            prevent_initial_call=True
        )
        def update_table_options(n_clicks, original_db, synthetic_db):
            """Update table selection dropdown options"""
            if not n_clicks:
                return []

            try:
                from ..core.database import get_database_manager
                db_manager = get_database_manager()

                # Get available tables from both databases
                tables_set = set()

                if original_db:
                    try:
                        original_tables = db_manager.get_table_list(original_db)
                        tables_set.update(original_tables)
                    except Exception as e:
                        self.logger.warning(f"Could not get tables from {original_db}: {e}")

                if synthetic_db and synthetic_db != original_db:
                    try:
                        synthetic_tables = db_manager.get_table_list(synthetic_db)
                        tables_set.update(synthetic_tables)
                    except Exception as e:
                        self.logger.warning(f"Could not get tables from {synthetic_db}: {e}")

                options = [{'label': table, 'value': table} for table in sorted(tables_set)]
                return options

            except Exception as e:
                self.logger.exception("Failed to update table options")
                return []


        @self.app.callback(
            Output("charts-modal", "is_open"),
            Output("comparison-charts-content", "children"),
            Input("show-charts-btn", "n_clicks"),
            Input("close-charts-btn", "n_clicks"),
            State("charts-modal", "is_open"),
            State('table-selection-dropdown', 'value'),
            State('original-db-input', 'value'),
            State('synthetic-db-input', 'value'),
            prevent_initial_call=True
        )
        def toggle_charts_modal(show_clicks, close_clicks, is_open, selected_tables, original_db, synthetic_db):
            """Toggle charts modal and generate comparison charts"""
            ctx = dash.callback_context
            if not ctx.triggered:
                return False, []

            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if trigger_id == "close-charts-btn":
                return False, []
            elif trigger_id == "show-charts-btn":
                if not selected_tables:
                    return False, [html.P("Please select tables first")]

                charts = self._generate_comparison_charts(selected_tables, original_db, synthetic_db)
                return True, charts

            return is_open, []

        @self.app.callback(
            Output("settings-modal", "is_open"),
            Input("settings-btn", "n_clicks"),
            Input("close-settings-btn", "n_clicks"),
            State("settings-modal", "is_open"),
            prevent_initial_call=True
        )
        def toggle_settings_modal(settings_clicks, close_clicks, is_open):
            """Toggle settings modal"""
            return not is_open

        @self.app.callback(
            Output("download-csv", "data"),
            Input("export-csv-btn", "n_clicks"),
            State('table-selection-dropdown', 'value'),
            State('comparison-type-radio', 'value'),
            State('original-db-input', 'value'),
            State('synthetic-db-input', 'value'),
            prevent_initial_call=True
        )
        def export_comparison_csv(n_clicks, selected_tables, comparison_type, original_db, synthetic_db):
            """Export comparison results to CSV"""
            if not n_clicks or not selected_tables:
                return dash.no_update

            self.logger.info(f"CSV export requested for tables: {selected_tables}")

            try:
                # Generate comparison data (similar to update_immediate_comparison)
                from ..core.database import get_database_manager
                db_manager = get_database_manager()

                comparison_data = []

                for table_name in selected_tables:
                    try:
                        original_info = db_manager.get_table_info(table_name, original_db)
                        synthetic_info = db_manager.get_table_info(table_name, synthetic_db)

                        if 'error' in original_info or 'error' in synthetic_info:
                            continue

                        original_sample = original_info.get('sample_data', pd.DataFrame())
                        synthetic_sample = synthetic_info.get('sample_data', pd.DataFrame())

                        if original_sample.empty or synthetic_sample.empty:
                            continue

                        common_columns = set(original_sample.columns) & set(synthetic_sample.columns)

                        for column in sorted(common_columns):
                            try:
                                orig_series = original_sample[column]
                                synth_series = synthetic_sample[column]

                                is_numeric_orig = pd.api.types.is_numeric_dtype(orig_series)
                                is_numeric_synth = pd.api.types.is_numeric_dtype(synth_series)

                                # Skip based on comparison type filter
                                if comparison_type == 'numeric' and not (is_numeric_orig and is_numeric_synth):
                                    continue
                                elif comparison_type == 'categorical' and (is_numeric_orig and is_numeric_synth):
                                    continue

                                column_comparison = {
                                    'table': table_name,
                                    'column': column,
                                    'data_type': 'numeric' if (is_numeric_orig and is_numeric_synth) else 'categorical',
                                    'original_dtype': str(orig_series.dtype),
                                    'synthetic_dtype': str(synth_series.dtype),
                                    'original_count': len(orig_series),
                                    'synthetic_count': len(synth_series),
                                    'original_missing': orig_series.isna().sum(),
                                    'synthetic_missing': synth_series.isna().sum()
                                }

                                if is_numeric_orig and is_numeric_synth:
                                    # Add numeric statistics
                                    orig_clean = orig_series.dropna()
                                    synth_clean = synth_series.dropna()

                                    if len(orig_clean) > 0 and len(synth_clean) > 0:
                                        column_comparison.update({
                                            'original_mean': float(orig_clean.mean()),
                                            'synthetic_mean': float(synth_clean.mean()),
                                            'original_median': float(orig_clean.median()),
                                            'synthetic_median': float(synth_clean.median()),
                                            'original_std': float(orig_clean.std()) if len(orig_clean) > 1 else 0.0,
                                            'synthetic_std': float(synth_clean.std()) if len(synth_clean) > 1 else 0.0,
                                            'original_min': float(orig_clean.min()),
                                            'synthetic_min': float(synth_clean.min()),
                                            'original_max': float(orig_clean.max()),
                                            'synthetic_max': float(synth_clean.max()),
                                            'original_q25': float(orig_clean.quantile(0.25)),
                                            'synthetic_q25': float(synth_clean.quantile(0.25)),
                                            'original_q75': float(orig_clean.quantile(0.75)),
                                            'synthetic_q75': float(synth_clean.quantile(0.75))
                                        })

                                        # Calculate difference percentages
                                        if column_comparison['original_mean'] != 0:
                                            column_comparison['mean_diff_pct'] = abs(
                                                (column_comparison['synthetic_mean'] - column_comparison['original_mean']) /
                                                column_comparison['original_mean'] * 100
                                            )
                                        else:
                                            column_comparison['mean_diff_pct'] = 0.0

                                        if column_comparison['original_std'] != 0:
                                            column_comparison['std_diff_pct'] = abs(
                                                (column_comparison['synthetic_std'] - column_comparison['original_std']) /
                                                column_comparison['original_std'] * 100
                                            )
                                        else:
                                            column_comparison['std_diff_pct'] = 0.0
                                else:
                                    # Add categorical statistics
                                    orig_counts = orig_series.value_counts()
                                    synth_counts = synth_series.value_counts()

                                    column_comparison.update({
                                        'original_unique': len(orig_counts),
                                        'synthetic_unique': len(synth_counts),
                                        'original_mode': str(orig_counts.index[0]) if len(orig_counts) > 0 else 'N/A',
                                        'synthetic_mode': str(synth_counts.index[0]) if len(synth_counts) > 0 else 'N/A',
                                        'original_mode_freq': int(orig_counts.iloc[0]) if len(orig_counts) > 0 else 0,
                                        'synthetic_mode_freq': int(synth_counts.iloc[0]) if len(synth_counts) > 0 else 0
                                    })

                                    if column_comparison['original_count'] > 0:
                                        column_comparison['original_mode_pct'] = (column_comparison['original_mode_freq'] / column_comparison['original_count']) * 100
                                    else:
                                        column_comparison['original_mode_pct'] = 0.0

                                    if column_comparison['synthetic_count'] > 0:
                                        column_comparison['synthetic_mode_pct'] = (column_comparison['synthetic_mode_freq'] / column_comparison['synthetic_count']) * 100
                                    else:
                                        column_comparison['synthetic_mode_pct'] = 0.0

                                comparison_data.append(column_comparison)

                            except Exception as e:
                                self.logger.warning(f"Failed to analyze column {column} in table {table_name}: {e}")
                                continue

                    except Exception as e:
                        self.logger.warning(f"Failed to analyze table {table_name}: {e}")
                        continue

                if comparison_data:
                    # Convert to DataFrame and generate CSV for download
                    df = pd.DataFrame(comparison_data)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"table_comparison_{comparison_type}_{timestamp}.csv"

                    # Convert to CSV string for download
                    csv_string = df.to_csv(index=False)
                    self.logger.info(f"CSV generated successfully: {filename} ({len(csv_string)} characters)")

                    return dict(content=csv_string, filename=filename)
                else:
                    self.logger.warning("No comparison data available for CSV export")
                    return dash.no_update

            except Exception as e:
                self.logger.exception("Failed to export comparison data")
                return dash.no_update

        @self.app.callback(
            Output("download-pdf", "data"),
            Output("charts-data", "data"),
            Input("export-pdf-btn", "n_clicks"),
            State('table-selection-dropdown', 'value'),
            State('original-db-input', 'value'),
            State('synthetic-db-input', 'value'),
            prevent_initial_call=True
        )
        def export_charts_pdf(n_clicks, selected_tables, original_db, synthetic_db):
            """Export comparison charts to PDF"""
            if not n_clicks or not selected_tables:
                return dash.no_update, dash.no_update

            self.logger.info(f"PDF export requested for tables: {selected_tables}")

            try:
                # Generate charts and export to PDF
                import io
                from reportlab.lib.pagesizes import A4
                from reportlab.lib.units import inch
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
                import tempfile
                import os

                # Create temporary directory for chart images
                temp_dir = tempfile.mkdtemp()
                self.logger.info(f"Created temp directory: {temp_dir}")

                try:
                    # Generate charts for the selected tables
                    charts = self._generate_comparison_charts(selected_tables, original_db, synthetic_db)
                    self.logger.info(f"Generated {len(charts)} chart components")

                    # Create PDF buffer
                    pdf_buffer = io.BytesIO()
                    doc = SimpleDocTemplate(pdf_buffer, pagesize=A4,
                                          topMargin=inch, bottomMargin=inch,
                                          leftMargin=inch, rightMargin=inch)
                    story = []

                    # Add title
                    styles = getSampleStyleSheet()
                    title = Paragraph("NEDIS Data Comparison Report", styles['Title'])
                    story.append(title)
                    story.append(Spacer(1, 20))

                    # Add metadata
                    metadata = Paragraph(f"""
                    <b>Tables Analyzed:</b> {', '.join(selected_tables)}<br/>
                    <b>Original Database:</b> {original_db or 'nedis_data.duckdb'}<br/>
                    <b>Synthetic Database:</b> {synthetic_db or 'nedis_synth_2017.duckdb'}<br/>
                    <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
                    """, styles['Normal'])
                    story.append(metadata)
                    story.append(Spacer(1, 20))

                    # Process each chart (Plotly figures) and save as images
                    chart_count = 0
                    for i, chart_component in enumerate(charts):
                        try:
                            if hasattr(chart_component, 'figure') and chart_component.figure:
                                # Export plotly figure to image
                                fig = chart_component.figure
                                img_path = os.path.join(temp_dir, f"chart_{i}.png")

                                # Use kaleido to export image
                                fig.write_image(img_path, width=800, height=600, engine="kaleido")
                                self.logger.info(f"Exported chart {i} to {img_path}")

                                # Add chart to PDF
                                if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                                    # Add chart title if available
                                    if (hasattr(fig, 'layout') and hasattr(fig.layout, 'title')
                                        and fig.layout.title and hasattr(fig.layout.title, 'text')
                                        and fig.layout.title.text):
                                        chart_title = Paragraph(f"<b>{fig.layout.title.text}</b>", styles['Heading2'])
                                        story.append(chart_title)
                                        story.append(Spacer(1, 10))

                                    # Add image to PDF
                                    img = Image(img_path, width=6*inch, height=4.5*inch)
                                    story.append(img)
                                    story.append(Spacer(1, 20))
                                    chart_count += 1
                                    self.logger.info(f"Added chart {i} to PDF")

                        except Exception as chart_error:
                            self.logger.warning(f"Failed to export chart {i}: {chart_error}")
                            continue

                    if chart_count == 0:
                        # Add message if no charts were generated
                        no_charts = Paragraph("No charts were available for PDF export. Please ensure you have run a comparison first.", styles['Normal'])
                        story.append(no_charts)
                        self.logger.warning("No charts were exported to PDF")

                    # Build PDF
                    doc.build(story)
                    pdf_buffer.seek(0)
                    pdf_data = pdf_buffer.getvalue()

                    # Clean up temp files
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)

                    # Return PDF for download
                    filename = f"nedis_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    self.logger.info(f"PDF generated successfully: {filename} ({len(pdf_data)} bytes)")

                    return dict(content=pdf_data, filename=filename, type="application/pdf"), {"exported": True, "charts": chart_count}

                except Exception as pdf_error:
                    # Clean up temp files on error
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    self.logger.error(f"PDF generation error: {pdf_error}")
                    raise pdf_error

            except Exception as e:
                self.logger.exception("Failed to export charts to PDF")
                # Return error state instead of None
                return dash.no_update, {"error": str(e)}

    def _start_background_updates(self):
        """Start background thread for live updates"""
        def background_update():
            self.logger.debug("Background update thread started")
            while True:
                try:
                    time.sleep(5)  # Update every 5 seconds
                except Exception as e:
                    self.logger.exception("Background update error")

        update_thread = threading.Thread(target=background_update, daemon=True)
        update_thread.start()

    def _start_validation_job(self, original_db: str, synthetic_db: str,
                              sample_size: Optional[int],
                              validation_type: str = 'comprehensive') -> Optional[str]:
        """Launch validation work in a background thread"""
        with self._state_lock:
            if self._current_job and self._current_job.is_alive():
                self._log_progress("Validation request ignored; a job is already running")
                return None

        validation_id = f"dashboard_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S_%f')}"

        def worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            formatted_samples = f"{sample_size:,}" if isinstance(sample_size, int) else 'default'
            self._log_progress(
                f"Validation job started (id={validation_id}): original='{original_db}', synthetic='{synthetic_db}', samples={formatted_samples}"
            )

            try:
                if validation_type != 'comprehensive':
                    raise ValueError(f"Unsupported validation type from dashboard: {validation_type}")

                result = loop.run_until_complete(
                    self.orchestrator.validate_comprehensive(original_db, synthetic_db, sample_size)
                )

                summary = self._summarize_result(result)
                self._last_metrics_snapshot = None  # force metrics log refresh
                score = summary.get('overall_score')
                duration = summary.get('duration_seconds')
                score_text = f"{score:.2f}" if isinstance(score, (int, float)) else 'N/A'
                duration_text = f"{duration:.1f}s" if isinstance(duration, (int, float)) else 'unknown duration'
                message = f"Validation completed (score={score_text}, duration={duration_text})"
                self._log_progress(message)
                self._queue_state_update({
                    'status': 'completed',
                    'message': message,
                    'progress': 100,
                    'result': summary,
                    'statistical_summary': summary.get('statistical_summary'),
                    'validation_id': validation_id,
                    'completed_at': datetime.now(timezone.utc).isoformat()
                })
            except Exception as exc:
                self.logger.exception("Validation job failed")
                self._queue_state_update({
                    'status': 'failed',
                    'message': f"Validation failed: {exc}",
                    'error': str(exc),
                    'progress': 0,
                    'validation_id': validation_id,
                    'completed_at': datetime.now(timezone.utc).isoformat()
                })
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)
                with self._state_lock:
                    self._current_job = None
                    self._active_validation_id = None

        worker_thread = threading.Thread(
            target=worker,
            name=f"dashboard-validation-{validation_id}",
            daemon=True
        )

        with self._state_lock:
            self._current_job = worker_thread
            self._active_validation_id = validation_id

        worker_thread.start()
        return validation_id

    def _queue_state_update(self, update: Dict[str, Any]):
        """Queue state changes from background threads for the next callback tick"""
        with self._state_lock:
            if self._pending_state_update:
                self._pending_state_update.update(update)
            else:
                self._pending_state_update = update

    def _consume_pending_state_update(self) -> Optional[Dict[str, Any]]:
        """Fetch and clear pending state updates"""
        with self._state_lock:
            update = self._pending_state_update
            self._pending_state_update = None
            return update

    def run(self, debug: bool = False):
        """Run the dashboard server"""
        print("🚀 Starting NEDIS Validation Dashboard...")
        print(f"📍 Dashboard will be available at: http://{self.host}:{self.port}")
        print(f"📊 Open your browser and navigate to the URL above")

        self.app.run(
            host=self.host,
            port=self.port,
            debug=debug
        )

    def _state_to_outputs(self, state: Dict[str, Any]):
        """Map internal validation state to UI outputs"""
        if state is None:
            state = {}

        status = state.get('status', 'ready')
        message = state.get('message', '')
        progress = state.get('progress', 0)

        if status == 'running':
            display_status = "🔄 Validating..."
            if not message:
                sample_size = state.get('sample_size')
                formatted_samples = f"{sample_size:,}" if isinstance(sample_size, int) else 'default'
                message = f"Running validation with {formatted_samples} samples..."
            progress = max(progress, 10)
        elif status == 'completed':
            display_status = "✅ Validation completed"
            progress = 100
            if not message:
                result = state.get('result', {})
                score = result.get('overall_score')
                duration = result.get('duration_seconds')
                if score is not None and duration is not None:
                    message = f"Score {score:.2f} / Duration {duration:.1f}s"
                else:
                    message = "Validation finished successfully."
        elif status == 'failed':
            display_status = "❌ Validation failed"
            progress = 0
            if not message:
                message = state.get('error', 'Validation encountered an error.')
        else:
            display_status = "Ready"
            message = ""
            progress = 0

        return display_status, message, progress, state

    def _summarize_result(self, result: Any) -> Dict[str, Any]:
        """Convert ValidationResult into a light-weight summary"""
        if result is None:
            return {}

        summary = {
            'validation_type': getattr(result, 'validation_type', None),
            'overall_score': getattr(result, 'overall_score', None),
            'duration_seconds': getattr(result, 'duration', None),
            'start_time': getattr(result, 'start_time', None).isoformat() if getattr(result, 'start_time', None) else None,
            'end_time': getattr(result, 'end_time', None).isoformat() if getattr(result, 'end_time', None) else None,
            'errors': len(getattr(result, 'errors', []) or []),
            'warnings': len(getattr(result, 'warnings', []) or [])
        }

        statistical_summary = self._extract_statistical_summary(result)
        if statistical_summary:
            summary['statistical_summary'] = statistical_summary

        return summary

    def _extract_statistical_summary(self, result: Any) -> Dict[str, Any]:
        """Build a digestible statistical comparison table"""
        try:
            statistical = getattr(result, 'results', {}).get('statistical')
        except AttributeError:
            statistical = None

        if not statistical:
            return {}

        summary: Dict[str, Any] = {
            'overall_score': float(statistical.get('overall_score', 0.0)) if isinstance(statistical.get('overall_score'), (int, float)) else statistical.get('overall_score'),
            'sample_sizes': statistical.get('sample_sizes', {})
        }

        continuous_comparisons: List[Dict[str, Any]] = []
        for var, data in (statistical.get('continuous_tests', {}) or {}).items():
            if not isinstance(data, dict) or not data.get('available'):
                continue

            stats_block = data.get('statistics', {})
            orig_stats = stats_block.get('original', {})
            synth_stats = stats_block.get('synthetic', {})

            try:
                orig_mean = float(orig_stats.get('mean', 0.0))
                synth_mean = float(synth_stats.get('mean', 0.0))
            except (TypeError, ValueError):
                continue

            delta_mean = synth_mean - orig_mean
            wasserstein = data.get('wasserstein_distance', {}).get('distance', 0.0)
            ks_p_value = data.get('ks_test', {}).get('p_value', 0.0)

            continuous_comparisons.append({
                'variable': var,
                'original_mean': round(orig_mean, 4),
                'synthetic_mean': round(synth_mean, 4),
                'delta_mean': round(delta_mean, 4),
                'wasserstein': round(float(wasserstein), 4) if isinstance(wasserstein, (int, float)) else wasserstein,
                'ks_p_value': round(float(ks_p_value), 4) if isinstance(ks_p_value, (int, float)) else ks_p_value,
                'score': round(float(data.get('score', 0.0)), 4) if isinstance(data.get('score'), (int, float)) else data.get('score')
            })

        continuous_comparisons.sort(key=lambda x: abs(x['delta_mean']), reverse=True)
        summary['continuous'] = continuous_comparisons

        categorical_comparisons: List[Dict[str, Any]] = []
        for var, data in (statistical.get('categorical_tests', {}) or {}).items():
            if not isinstance(data, dict) or not data.get('available'):
                continue

            distributions = data.get('distributions', {})
            orig_dist = distributions.get('original', {}) or {}
            synth_dist = distributions.get('synthetic', {}) or {}

            if not orig_dist and not synth_dist:
                continue

            best_category = None
            best_delta = 0.0
            best_orig_pct = 0.0
            best_synth_pct = 0.0

            categories = set(orig_dist.keys()) | set(synth_dist.keys())
            for category in categories:
                orig_pct = float(orig_dist.get(category, 0.0)) * 100.0
                synth_pct = float(synth_dist.get(category, 0.0)) * 100.0
                delta_pct = synth_pct - orig_pct

                if best_category is None or abs(delta_pct) > abs(best_delta):
                    best_category = category
                    best_delta = delta_pct
                    best_orig_pct = orig_pct
                    best_synth_pct = synth_pct

            if best_category is None:
                continue

            chi2_p_value = data.get('chi2_test', {}).get('p_value', 0.0)

            categorical_comparisons.append({
                'variable': var,
                'category': str(best_category),
                'original_pct': round(best_orig_pct, 2),
                'synthetic_pct': round(best_synth_pct, 2),
                'delta_pct': round(best_delta, 2),
                'chi2_p_value': round(float(chi2_p_value), 4) if isinstance(chi2_p_value, (int, float)) else chi2_p_value,
                'score': round(float(data.get('score', 0.0)), 4) if isinstance(data.get('score'), (int, float)) else data.get('score')
            })

        categorical_comparisons.sort(key=lambda x: abs(x['delta_pct']), reverse=True)
        summary['categorical'] = categorical_comparisons

        column_profiles = statistical.get('column_profiles')
        if column_profiles:
            summary['column_profiles'] = column_profiles

        return summary

    def _log_progress(self, message: str):
        """Log progress updates to the console without spamming duplicates"""
        if message != self._last_progress_log:
            self.logger.info(message)
            self._last_progress_log = message

    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics changes only when something significant updates"""
        if metrics != self._last_metrics_snapshot:
            formatted = ", ".join(f"{key}={value}" for key, value in metrics.items())
            self.logger.info(f"Metrics update: {formatted}")
            self._last_metrics_snapshot = metrics

    def _build_table(self, rows: List[Dict[str, Any]], columns: List[str], labels: Dict[str, str]) -> Any:
        """Create a Bootstrap table from row data"""
        header_cells = [html.Th(labels.get(col, col)) for col in columns]
        header = html.Thead(html.Tr(header_cells))

        body_rows = []
        for row in rows:
            cells = []
            for col in columns:
                value = row.get(col, '')
                if isinstance(value, float):
                    value = f"{value:.4f}" if abs(value) < 1000 else f"{value:.2f}"
                cells.append(html.Td(value))
            body_rows.append(html.Tr(cells))

        body = html.Tbody(body_rows)

        return dbc.Table(
            [header, body],
            bordered=True,
            hover=True,
            responsive=True,
            striped=True,
            size="sm"
        )

    def _build_enhanced_comparison_table(self, data: List[Dict[str, Any]], comparison_type: str) -> List[Any]:
        """Build enhanced comparison table with color coding and additional statistics"""
        content = []

        if comparison_type in ['all', 'numeric']:
            numeric_data = [row for row in data if row.get('data_type') == 'numeric']
            if numeric_data:
                content.append(html.H5("📊 Numeric Columns", className="mt-3"))
                content.append(self._build_numeric_comparison_table(numeric_data))

        if comparison_type in ['all', 'categorical']:
            categorical_data = [row for row in data if row.get('data_type') == 'categorical']
            if categorical_data:
                content.append(html.H5("🏷️ Categorical Columns", className="mt-4"))
                content.append(self._build_categorical_comparison_table(categorical_data))

        # Summary statistics
        content.append(self._build_comparison_summary(data))

        return content

    def _build_numeric_comparison_table(self, data: List[Dict[str, Any]]) -> Any:
        """Build numeric comparison table with enhanced statistics and color coding"""
        header_cells = [
            html.Th("Table.Column"),
            html.Th("Data Type"),
            html.Th("Count (Orig → Synth)"),
            html.Th("Missing"),
            html.Th("Mean (Δ%)"),
            html.Th("Median"),
            html.Th("Std Dev (Δ%)"),
            html.Th("Min → Max"),
            html.Th("IQR"),
            html.Th("Skewness"),
            html.Th("Kurtosis"),
            html.Th("Range [5%, 95%]")
        ]
        header = html.Thead(html.Tr(header_cells))

        body_rows = []
        for row in data:
            # Color coding based on difference percentages
            mean_diff = row.get('mean_diff_pct', 0)
            std_diff = row.get('std_diff_pct', 0)

            mean_color = self._get_difference_color(mean_diff)
            std_color = self._get_difference_color(std_diff)

            cells = [
                html.Td(f"{row['table']}.{row['column']}", className="fw-bold"),
                html.Td(f"{row['original_dtype']} → {row['synthetic_dtype']}"),
                html.Td(f"{row['original_count']:,} → {row['synthetic_count']:,}"),
                html.Td(f"{row['original_missing']} → {row['synthetic_missing']}"),
                html.Td([
                    f"{row['original_mean']:.3f} → {row['synthetic_mean']:.3f}",
                    html.Br(),
                    html.Small(f"(Δ{mean_diff:.1f}%)", className=f"text-{mean_color}")
                ]),
                html.Td(f"{row['original_median']:.3f} → {row['synthetic_median']:.3f}"),
                html.Td([
                    f"{row['original_std']:.3f} → {row['synthetic_std']:.3f}",
                    html.Br(),
                    html.Small(f"(Δ{std_diff:.1f}%)", className=f"text-{std_color}")
                ]),
                html.Td(f"[{row['original_min']:.2f}, {row['original_max']:.2f}] → [{row['synthetic_min']:.2f}, {row['synthetic_max']:.2f}]"),
                html.Td(f"{row['original_iqr']:.3f} → {row['synthetic_iqr']:.3f}"),
                html.Td(f"{row.get('original_skewness', 0):.2f} → {row.get('synthetic_skewness', 0):.2f}"),
                html.Td(f"{row.get('original_kurtosis', 0):.2f} → {row.get('synthetic_kurtosis', 0):.2f}"),
                html.Td(f"[{row['original_q05']:.2f}, {row['original_q95']:.2f}] → [{row['synthetic_q05']:.2f}, {row['synthetic_q95']:.2f}]")
            ]
            body_rows.append(html.Tr(cells))

        body = html.Tbody(body_rows)

        return dbc.Table(
            [header, body],
            bordered=True,
            hover=True,
            responsive=True,
            striped=True,
            size="sm"
        )

    def _build_categorical_comparison_table(self, data: List[Dict[str, Any]]) -> Any:
        """Build categorical comparison table with enhanced statistics"""
        header_cells = [
            html.Th("Table.Column"),
            html.Th("Data Type"),
            html.Th("Count (Orig → Synth)"),
            html.Th("Missing"),
            html.Th("Unique Values (Δ%)"),
            html.Th("Most Frequent Value"),
            html.Th("Mode Frequency (%)")
        ]
        header = html.Thead(html.Tr(header_cells))

        body_rows = []
        for row in data:
            unique_diff = row.get('unique_diff_pct', 0)
            unique_color = self._get_difference_color(unique_diff)

            cells = [
                html.Td(f"{row['table']}.{row['column']}", className="fw-bold"),
                html.Td(f"{row['original_dtype']} → {row['synthetic_dtype']}"),
                html.Td(f"{row['original_count']:,} → {row['synthetic_count']:,}"),
                html.Td(f"{row['original_missing']} → {row['synthetic_missing']}"),
                html.Td([
                    f"{row['original_unique']} → {row['synthetic_unique']}",
                    html.Br(),
                    html.Small(f"(Δ{unique_diff:.1f}%)", className=f"text-{unique_color}")
                ]),
                html.Td(f"{row['original_mode']} → {row['synthetic_mode']}"),
                html.Td(f"{row['original_mode_freq']} ({row['original_mode_pct']:.1f}%) → {row['synthetic_mode_freq']} ({row['synthetic_mode_pct']:.1f}%)")
            ]
            body_rows.append(html.Tr(cells))

        body = html.Tbody(body_rows)

        return dbc.Table(
            [header, body],
            bordered=True,
            hover=True,
            responsive=True,
            striped=True,
            size="sm"
        )

    def _build_comparison_summary(self, data: List[Dict[str, Any]]) -> Any:
        """Build comparison summary with key metrics"""
        total_columns = len(data)
        numeric_columns = len([row for row in data if row.get('data_type') == 'numeric'])
        categorical_columns = len([row for row in data if row.get('data_type') == 'categorical'])

        # Calculate average differences for numeric columns
        numeric_data = [row for row in data if row.get('data_type') == 'numeric']
        avg_mean_diff = np.mean([row.get('mean_diff_pct', 0) for row in numeric_data]) if numeric_data else 0
        avg_std_diff = np.mean([row.get('std_diff_pct', 0) for row in numeric_data]) if numeric_data else 0

        # Calculate average differences for categorical columns
        categorical_data = [row for row in data if row.get('data_type') == 'categorical']
        avg_unique_diff = np.mean([row.get('unique_diff_pct', 0) for row in categorical_data]) if categorical_data else 0

        return dbc.Card([
            dbc.CardHeader("📈 Comparison Summary"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Total Columns Compared", className="text-muted"),
                        html.H4(f"{total_columns}", className="text-primary")
                    ], width=3),
                    dbc.Col([
                        html.H6("Numeric Columns", className="text-muted"),
                        html.H4(f"{numeric_columns}", className="text-info")
                    ], width=3),
                    dbc.Col([
                        html.H6("Categorical Columns", className="text-muted"),
                        html.H4(f"{categorical_columns}", className="text-success")
                    ], width=3),
                    dbc.Col([
                        html.H6("Overall Similarity", className="text-muted"),
                        html.H4(f"{max(0, 100 - (avg_mean_diff + avg_std_diff + avg_unique_diff) / 3):.1f}%",
                               className="text-warning")
                    ], width=3)
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.P([
                            html.Strong("Avg Mean Difference: "),
                            f"{avg_mean_diff:.1f}%"
                        ])
                    ], width=4),
                    dbc.Col([
                        html.P([
                            html.Strong("Avg Std Dev Difference: "),
                            f"{avg_std_diff:.1f}%"
                        ])
                    ], width=4),
                    dbc.Col([
                        html.P([
                            html.Strong("Avg Unique Count Difference: "),
                            f"{avg_unique_diff:.1f}%"
                        ])
                    ], width=4)
                ])
            ])
        ], className="mt-4")

    def _get_difference_color(self, diff_pct: float) -> str:
        """Get color class based on difference percentage"""
        if diff_pct <= 5:
            return "success"  # Green for small differences
        elif diff_pct <= 15:
            return "warning"  # Yellow for medium differences
        else:
            return "danger"   # Red for large differences

    def _generate_comparison_charts(self, selected_tables: List[str], original_db: str, synthetic_db: str) -> List[Any]:
        """Generate comprehensive comparison charts for selected tables"""
        try:
            from ..core.database import get_database_manager
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            db_manager = get_database_manager()
            charts = []

            for table_name in selected_tables[:2]:  # Limit to 2 tables to avoid overcrowding
                try:
                    # Get table info
                    original_info = db_manager.get_table_info(table_name, original_db)
                    synthetic_info = db_manager.get_table_info(table_name, synthetic_db)

                    if 'error' in original_info or 'error' in synthetic_info:
                        continue

                    original_sample = original_info.get('sample_data', pd.DataFrame())
                    synthetic_sample = synthetic_info.get('sample_data', pd.DataFrame())

                    if original_sample.empty or synthetic_sample.empty:
                        continue

                    # Get numeric columns for visualizations
                    numeric_columns = []
                    for col in original_sample.columns:
                        if col in synthetic_sample.columns:
                            if pd.api.types.is_numeric_dtype(original_sample[col]) and pd.api.types.is_numeric_dtype(synthetic_sample[col]):
                                numeric_columns.append(col)

                    if numeric_columns:
                        # 1. HISTOGRAM OVERLAY COMPARISON
                        cols_to_plot = numeric_columns[:4]
                        hist_fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=[f"Distribution: {col}" for col in cols_to_plot],
                            vertical_spacing=0.15,
                            horizontal_spacing=0.1
                        )

                        for idx, col in enumerate(cols_to_plot):
                            row = (idx // 2) + 1
                            col_pos = (idx % 2) + 1

                            orig_data = original_sample[col].dropna()
                            synth_data = synthetic_sample[col].dropna()

                            # Filter out extreme outliers for better visualization
                            orig_q95 = orig_data.quantile(0.95) if len(orig_data) > 0 else float('inf')
                            orig_q05 = orig_data.quantile(0.05) if len(orig_data) > 0 else float('-inf')
                            synth_q95 = synth_data.quantile(0.95) if len(synth_data) > 0 else float('inf')
                            synth_q05 = synth_data.quantile(0.05) if len(synth_data) > 0 else float('-inf')

                            # Use common range for both datasets
                            min_val = min(orig_q05, synth_q05)
                            max_val = max(orig_q95, synth_q95)

                            if np.isfinite(min_val) and np.isfinite(max_val) and min_val < max_val:
                                orig_filtered = orig_data[(orig_data >= min_val) & (orig_data <= max_val)]
                                synth_filtered = synth_data[(synth_data >= min_val) & (synth_data <= max_val)]
                            else:
                                orig_filtered = orig_data
                                synth_filtered = synth_data

                            # Add overlapping histograms
                            hist_fig.add_trace(
                                go.Histogram(x=orig_filtered, name=f"Original",
                                           opacity=0.6, nbinsx=25, histnorm='probability',
                                           marker_color='blue', legendgroup=f'{col}'),
                                row=row, col=col_pos
                            )
                            hist_fig.add_trace(
                                go.Histogram(x=synth_filtered, name=f"Synthetic",
                                           opacity=0.6, nbinsx=25, histnorm='probability',
                                           marker_color='red', legendgroup=f'{col}'),
                                row=row, col=col_pos
                            )

                        hist_fig.update_layout(
                            title=f"📊 Distribution Overlay: {table_name}",
                            height=600,
                            showlegend=True,
                            barmode='overlay',
                            font=dict(size=10)
                        )
                        charts.append(dcc.Graph(figure=hist_fig))

                        # 2. BOX PLOT COMPARISON
                        if len(numeric_columns) > 0:
                            box_cols = numeric_columns[:6]  # Up to 6 columns for box plots
                            box_fig = go.Figure()

                            for col in box_cols:
                                orig_data = original_sample[col].dropna()
                                synth_data = synthetic_sample[col].dropna()

                                # Add box plots side by side
                                box_fig.add_trace(go.Box(
                                    y=orig_data,
                                    name=f"Original {col}",
                                    boxpoints='outliers',
                                    marker_color='blue',
                                    x=[f"{col}_Original"] * len(orig_data)
                                ))

                                box_fig.add_trace(go.Box(
                                    y=synth_data,
                                    name=f"Synthetic {col}",
                                    boxpoints='outliers',
                                    marker_color='red',
                                    x=[f"{col}_Synthetic"] * len(synth_data)
                                ))

                            box_fig.update_layout(
                                title=f"📦 Box Plot Comparison: {table_name}",
                                height=500,
                                showlegend=True,
                                xaxis_title="Columns",
                                yaxis_title="Values",
                                boxmode='group'
                            )
                            charts.append(dcc.Graph(figure=box_fig))

                        # 3. VIOLIN PLOT COMPARISON
                        if len(numeric_columns) > 0:
                            violin_cols = numeric_columns[:4]
                            violin_fig = make_subplots(
                                rows=2, cols=2,
                                subplot_titles=[f"Violin: {col}" for col in violin_cols],
                                vertical_spacing=0.15,
                                horizontal_spacing=0.1
                            )

                            for idx, col in enumerate(violin_cols):
                                row = (idx // 2) + 1
                                col_pos = (idx % 2) + 1

                                orig_data = original_sample[col].dropna()
                                synth_data = synthetic_sample[col].dropna()

                                # Add violin plots
                                violin_fig.add_trace(go.Violin(
                                    y=orig_data,
                                    name=f"Original {col}",
                                    side='negative',
                                    line_color='blue',
                                    fillcolor='lightblue',
                                    opacity=0.6,
                                    x0=col
                                ), row=row, col=col_pos)

                                violin_fig.add_trace(go.Violin(
                                    y=synth_data,
                                    name=f"Synthetic {col}",
                                    side='positive',
                                    line_color='red',
                                    fillcolor='lightcoral',
                                    opacity=0.6,
                                    x0=col
                                ), row=row, col=col_pos)

                            violin_fig.update_layout(
                                title=f"🎻 Violin Plot Comparison: {table_name}",
                                height=600,
                                showlegend=True,
                                violinmode='overlay'
                            )
                            charts.append(dcc.Graph(figure=violin_fig))

                        # 4. Q-Q PLOT COMPARISON
                        if len(numeric_columns) > 0:
                            qq_cols = numeric_columns[:4]
                            qq_fig = make_subplots(
                                rows=2, cols=2,
                                subplot_titles=[f"Q-Q Plot: {col}" for col in qq_cols],
                                vertical_spacing=0.15,
                                horizontal_spacing=0.1
                            )

                            for idx, col in enumerate(qq_cols):
                                row = (idx // 2) + 1
                                col_pos = (idx % 2) + 1

                                orig_data = original_sample[col].dropna()
                                synth_data = synthetic_sample[col].dropna()

                                if len(orig_data) > 10 and len(synth_data) > 10:
                                    # Create Q-Q plot data
                                    n_points = min(len(orig_data), len(synth_data), 100)
                                    orig_quantiles = np.quantile(orig_data, np.linspace(0.01, 0.99, n_points))
                                    synth_quantiles = np.quantile(synth_data, np.linspace(0.01, 0.99, n_points))

                                    # Add Q-Q scatter plot
                                    qq_fig.add_trace(go.Scatter(
                                        x=orig_quantiles,
                                        y=synth_quantiles,
                                        mode='markers',
                                        name=f"Q-Q {col}",
                                        marker=dict(color='green', opacity=0.6)
                                    ), row=row, col=col_pos)

                                    # Add diagonal line (perfect match)
                                    min_val = min(orig_quantiles.min(), synth_quantiles.min())
                                    max_val = max(orig_quantiles.max(), synth_quantiles.max())
                                    qq_fig.add_trace(go.Scatter(
                                        x=[min_val, max_val],
                                        y=[min_val, max_val],
                                        mode='lines',
                                        name=f"Perfect Match",
                                        line=dict(color='red', dash='dash'),
                                        showlegend=(idx == 0)
                                    ), row=row, col=col_pos)

                            qq_fig.update_layout(
                                title=f"📈 Q-Q Plot Comparison: {table_name}",
                                height=600,
                                showlegend=True
                            )
                            qq_fig.update_xaxes(title_text="Original Quantiles")
                            qq_fig.update_yaxes(title_text="Synthetic Quantiles")
                            charts.append(dcc.Graph(figure=qq_fig))

                        # 5. STATISTICAL SUMMARY HEATMAP
                        if len(numeric_columns) >= 2:
                            summary_cols = numeric_columns[:8]
                            stats_data = []

                            for col in summary_cols:
                                orig_data = original_sample[col].dropna()
                                synth_data = synthetic_sample[col].dropna()

                                if len(orig_data) > 0 and len(synth_data) > 0:
                                    orig_stats = [orig_data.mean(), orig_data.std(), orig_data.median(),
                                                 orig_data.quantile(0.25), orig_data.quantile(0.75)]
                                    synth_stats = [synth_data.mean(), synth_data.std(), synth_data.median(),
                                                  synth_data.quantile(0.25), synth_data.quantile(0.75)]

                                    # Calculate percentage differences
                                    pct_diffs = []
                                    for o, s in zip(orig_stats, synth_stats):
                                        if o != 0:
                                            pct_diffs.append(abs((s - o) / o * 100))
                                        else:
                                            pct_diffs.append(0)

                                    stats_data.append(pct_diffs)

                            if stats_data:
                                heatmap_fig = go.Figure(data=go.Heatmap(
                                    z=stats_data,
                                    x=['Mean', 'Std', 'Median', 'Q1', 'Q3'],
                                    y=summary_cols,
                                    colorscale='RdYlGn_r',
                                    colorbar=dict(title="% Difference"),
                                    text=[[f"{val:.1f}%" for val in row] for row in stats_data],
                                    texttemplate="%{text}",
                                    textfont={"size": 10}
                                ))

                                heatmap_fig.update_layout(
                                    title=f"🔥 Statistical Differences Heatmap: {table_name}",
                                    height=400,
                                    xaxis_title="Statistics",
                                    yaxis_title="Columns"
                                )
                                charts.append(dcc.Graph(figure=heatmap_fig))

                except Exception as e:
                    self.logger.warning(f"Failed to generate charts for table {table_name}: {e}")
                    continue

            if not charts:
                charts.append(html.P("No charts could be generated for the selected tables.",
                                   style={'text-align': 'center', 'color': 'gray', 'padding': '20px'}))

            return charts

        except Exception as e:
            self.logger.exception("Failed to generate comparison charts")
            return [html.P(f"Error generating charts: {str(e)}",
                          style={'color': 'red', 'text-align': 'center', 'padding': '20px'})]

    def _generate_immediate_comparison(self, selected_tables: List[str], comparison_type: str, original_db: str, synthetic_db: str) -> List[Any]:
        """Generate immediate comparison without validation"""
        try:
            from ..core.database import get_database_manager
            db_manager = get_database_manager()

            comparison_data = []

            for table_name in selected_tables:
                try:
                    # Get table info from both databases
                    original_info = db_manager.get_table_info(table_name, original_db)
                    synthetic_info = db_manager.get_table_info(table_name, synthetic_db)

                    if 'error' in original_info or 'error' in synthetic_info:
                        continue

                    # Get sample data for analysis
                    original_sample = original_info.get('sample_data', pd.DataFrame())
                    synthetic_sample = synthetic_info.get('sample_data', pd.DataFrame())

                    if original_sample.empty or synthetic_sample.empty:
                        continue

                    # Get common columns
                    common_columns = set(original_sample.columns) & set(synthetic_sample.columns)

                    for column in sorted(common_columns):
                        try:
                            orig_series = original_sample[column]
                            synth_series = synthetic_sample[column]

                            # Determine if numeric or categorical
                            is_numeric_orig = pd.api.types.is_numeric_dtype(orig_series)
                            is_numeric_synth = pd.api.types.is_numeric_dtype(synth_series)

                            # Skip based on comparison type filter
                            if comparison_type == 'numeric' and not (is_numeric_orig and is_numeric_synth):
                                continue
                            elif comparison_type == 'categorical' and (is_numeric_orig and is_numeric_synth):
                                continue

                            # Basic column info
                            column_comparison = {
                                'table': table_name,
                                'column': column,
                                'original_dtype': str(orig_series.dtype),
                                'synthetic_dtype': str(synth_series.dtype),
                                'original_count': len(orig_series),
                                'synthetic_count': len(synth_series),
                                'original_missing': orig_series.isna().sum(),
                                'synthetic_missing': synth_series.isna().sum()
                            }

                            if is_numeric_orig and is_numeric_synth:
                                # Enhanced numeric statistics
                                orig_clean = orig_series.dropna()
                                synth_clean = synth_series.dropna()

                                if len(orig_clean) > 0 and len(synth_clean) > 0:
                                    column_comparison.update({
                                        'data_type': 'numeric',
                                        'original_mean': float(orig_clean.mean()),
                                        'synthetic_mean': float(synth_clean.mean()),
                                        'original_median': float(orig_clean.median()),
                                        'synthetic_median': float(synth_clean.median()),
                                        'original_std': float(orig_clean.std()) if len(orig_clean) > 1 else 0.0,
                                        'synthetic_std': float(synth_clean.std()) if len(synth_clean) > 1 else 0.0,
                                        'original_min': float(orig_clean.min()),
                                        'synthetic_min': float(synth_clean.min()),
                                        'original_max': float(orig_clean.max()),
                                        'synthetic_max': float(synth_clean.max()),
                                        'original_q25': float(orig_clean.quantile(0.25)),
                                        'synthetic_q25': float(synth_clean.quantile(0.25)),
                                        'original_q75': float(orig_clean.quantile(0.75)),
                                        'synthetic_q75': float(synth_clean.quantile(0.75)),
                                        'original_q05': float(orig_clean.quantile(0.05)),
                                        'synthetic_q05': float(synth_clean.quantile(0.05)),
                                        'original_q95': float(orig_clean.quantile(0.95)),
                                        'synthetic_q95': float(synth_clean.quantile(0.95))
                                    })

                                    # Calculate additional statistics
                                    orig_iqr = column_comparison['original_q75'] - column_comparison['original_q25']
                                    synth_iqr = column_comparison['synthetic_q75'] - column_comparison['synthetic_q25']
                                    column_comparison['original_iqr'] = orig_iqr
                                    column_comparison['synthetic_iqr'] = synth_iqr

                                    # Calculate skewness and kurtosis
                                    try:
                                        from scipy import stats as scipy_stats
                                        column_comparison['original_skewness'] = float(scipy_stats.skew(orig_clean))
                                        column_comparison['synthetic_skewness'] = float(scipy_stats.skew(synth_clean))
                                        column_comparison['original_kurtosis'] = float(scipy_stats.kurtosis(orig_clean))
                                        column_comparison['synthetic_kurtosis'] = float(scipy_stats.kurtosis(synth_clean))
                                    except ImportError:
                                        column_comparison['original_skewness'] = 0.0
                                        column_comparison['synthetic_skewness'] = 0.0
                                        column_comparison['original_kurtosis'] = 0.0
                                        column_comparison['synthetic_kurtosis'] = 0.0

                                    # Calculate difference percentages
                                    if column_comparison['original_mean'] != 0:
                                        column_comparison['mean_diff_pct'] = abs(
                                            (column_comparison['synthetic_mean'] - column_comparison['original_mean']) /
                                            column_comparison['original_mean'] * 100
                                        )
                                    else:
                                        column_comparison['mean_diff_pct'] = 0.0

                                    if column_comparison['original_std'] != 0:
                                        column_comparison['std_diff_pct'] = abs(
                                            (column_comparison['synthetic_std'] - column_comparison['original_std']) /
                                            column_comparison['original_std'] * 100
                                        )
                                    else:
                                        column_comparison['std_diff_pct'] = 0.0
                            else:
                                # Categorical statistics
                                orig_counts = orig_series.value_counts()
                                synth_counts = synth_series.value_counts()

                                column_comparison.update({
                                    'data_type': 'categorical',
                                    'original_unique': len(orig_counts),
                                    'synthetic_unique': len(synth_counts),
                                    'original_mode': str(orig_counts.index[0]) if len(orig_counts) > 0 else 'N/A',
                                    'synthetic_mode': str(synth_counts.index[0]) if len(synth_counts) > 0 else 'N/A',
                                    'original_mode_freq': int(orig_counts.iloc[0]) if len(orig_counts) > 0 else 0,
                                    'synthetic_mode_freq': int(synth_counts.iloc[0]) if len(synth_counts) > 0 else 0
                                })

                                # Calculate mode percentages and differences
                                if column_comparison['original_count'] > 0:
                                    column_comparison['original_mode_pct'] = (column_comparison['original_mode_freq'] / column_comparison['original_count']) * 100
                                else:
                                    column_comparison['original_mode_pct'] = 0.0

                                if column_comparison['synthetic_count'] > 0:
                                    column_comparison['synthetic_mode_pct'] = (column_comparison['synthetic_mode_freq'] / column_comparison['synthetic_count']) * 100
                                else:
                                    column_comparison['synthetic_mode_pct'] = 0.0

                                # Calculate unique count difference
                                if column_comparison['original_unique'] != 0:
                                    column_comparison['unique_diff_pct'] = abs(
                                        (column_comparison['synthetic_unique'] - column_comparison['original_unique']) /
                                        column_comparison['original_unique'] * 100
                                    )
                                else:
                                    column_comparison['unique_diff_pct'] = 0.0

                            comparison_data.append(column_comparison)

                        except Exception as e:
                            self.logger.warning(f"Failed to analyze column {column} in table {table_name}: {e}")
                            continue

                except Exception as e:
                    self.logger.warning(f"Failed to analyze table {table_name}: {e}")
                    continue

            if not comparison_data:
                return html.P("No column data could be analyzed from selected tables.")

            # Filter data based on comparison type
            if comparison_type == 'numeric':
                filtered_data = [row for row in comparison_data if row.get('data_type') == 'numeric']
            elif comparison_type == 'categorical':
                filtered_data = [row for row in comparison_data if row.get('data_type') == 'categorical']
            else:
                filtered_data = comparison_data

            if not filtered_data:
                return html.P(f"No {comparison_type} columns found in selected tables.")

            # Generate enhanced comparison tables
            return self._build_enhanced_comparison_table(filtered_data, comparison_type)

        except Exception as e:
            self.logger.exception("Failed to generate immediate comparison")
            return html.P(f"Error: {str(e)}")

    def _generate_validation_comparison(self, original_db: str, synthetic_db: str) -> List[Any]:
        """Generate comparison from validation state (existing functionality)"""
        try:
            from ..core.database import get_database_manager
            db_manager = get_database_manager()

            # Get table lists from both databases
            original_tables = db_manager.get_table_list(original_db)
            synthetic_tables = db_manager.get_table_list(synthetic_db)

            if not original_tables:
                return html.P("Could not retrieve table information from original database.")

            # For demo purposes, if synthetic database has no meaningful tables,
            # compare original database to itself
            if not synthetic_tables or (len(synthetic_tables) == 1 and synthetic_tables[0] == 'test'):
                synthetic_db = original_db
                synthetic_tables = original_tables
                return html.P("Synthetic database appears to be empty or contains only test data. Comparing original database to itself for demonstration purposes.")

            # Get common tables
            common_tables = set(original_tables) & set(synthetic_tables)

            if not common_tables:
                return html.P("No common tables found between databases.")

            # Build comparison data for all tables (basic version for validation)
            comparison_data = []

            for table_name in sorted(common_tables):
                try:
                    # Get table info from both databases
                    original_info = db_manager.get_table_info(table_name, original_db)
                    synthetic_info = db_manager.get_table_info(table_name, synthetic_db)

                    if 'error' in original_info or 'error' in synthetic_info:
                        continue

                    # Get sample data for analysis
                    original_sample = original_info.get('sample_data', pd.DataFrame())
                    synthetic_sample = synthetic_info.get('sample_data', pd.DataFrame())

                    if original_sample.empty or synthetic_sample.empty:
                        continue

                    # Get common columns
                    common_columns = set(original_sample.columns) & set(synthetic_sample.columns)

                    for column in sorted(common_columns):
                        try:
                            orig_series = original_sample[column]
                            synth_series = synthetic_sample[column]

                            # Basic column info
                            column_comparison = {
                                'table': table_name,
                                'column': column,
                                'original_dtype': str(orig_series.dtype),
                                'synthetic_dtype': str(synth_series.dtype),
                                'original_count': len(orig_series),
                                'synthetic_count': len(synth_series),
                                'original_missing': orig_series.isna().sum(),
                                'synthetic_missing': synth_series.isna().sum()
                            }

                            # Determine if numeric or categorical
                            is_numeric_orig = pd.api.types.is_numeric_dtype(orig_series)
                            is_numeric_synth = pd.api.types.is_numeric_dtype(synth_series)

                            if is_numeric_orig and is_numeric_synth:
                                # Numeric statistics (basic version)
                                orig_clean = orig_series.dropna()
                                synth_clean = synth_series.dropna()

                                if len(orig_clean) > 0 and len(synth_clean) > 0:
                                    column_comparison.update({
                                        'data_type': 'numeric',
                                        'original_mean': float(orig_clean.mean()),
                                        'synthetic_mean': float(synth_clean.mean()),
                                        'original_median': float(orig_clean.median()),
                                        'synthetic_median': float(synth_clean.median()),
                                        'original_std': float(orig_clean.std()) if len(orig_clean) > 1 else 0.0,
                                        'synthetic_std': float(synth_clean.std()) if len(synth_clean) > 1 else 0.0,
                                        'original_min': float(orig_clean.min()),
                                        'synthetic_min': float(synth_clean.min()),
                                        'original_max': float(orig_clean.max()),
                                        'synthetic_max': float(synth_clean.max()),
                                        'original_q25': float(orig_clean.quantile(0.25)),
                                        'synthetic_q25': float(synth_clean.quantile(0.25)),
                                        'original_q75': float(orig_clean.quantile(0.75)),
                                        'synthetic_q75': float(synth_clean.quantile(0.75))
                                    })

                                    # Calculate IQR
                                    orig_iqr = column_comparison['original_q75'] - column_comparison['original_q25']
                                    synth_iqr = column_comparison['synthetic_q75'] - column_comparison['synthetic_q25']
                                    column_comparison['original_iqr'] = orig_iqr
                                    column_comparison['synthetic_iqr'] = synth_iqr
                            else:
                                # Categorical statistics
                                orig_counts = orig_series.value_counts()
                                synth_counts = synth_series.value_counts()

                                column_comparison.update({
                                    'data_type': 'categorical',
                                    'original_unique': len(orig_counts),
                                    'synthetic_unique': len(synth_counts),
                                    'original_mode': str(orig_counts.index[0]) if len(orig_counts) > 0 else 'N/A',
                                    'synthetic_mode': str(synth_counts.index[0]) if len(synth_counts) > 0 else 'N/A',
                                    'original_mode_freq': int(orig_counts.iloc[0]) if len(orig_counts) > 0 else 0,
                                    'synthetic_mode_freq': int(synth_counts.iloc[0]) if len(synth_counts) > 0 else 0
                                })

                                # Calculate mode percentages
                                if column_comparison['original_count'] > 0:
                                    column_comparison['original_mode_pct'] = (column_comparison['original_mode_freq'] / column_comparison['original_count']) * 100
                                else:
                                    column_comparison['original_mode_pct'] = 0.0

                                if column_comparison['synthetic_count'] > 0:
                                    column_comparison['synthetic_mode_pct'] = (column_comparison['synthetic_mode_freq'] / column_comparison['synthetic_count']) * 100
                                else:
                                    column_comparison['synthetic_mode_pct'] = 0.0

                            comparison_data.append(column_comparison)

                        except Exception as e:
                            self.logger.warning(f"Failed to analyze column {column} in table {table_name}: {e}")
                            continue

                except Exception as e:
                    self.logger.warning(f"Failed to analyze table {table_name}: {e}")
                    continue

            if not comparison_data:
                return html.P("No column data could be analyzed.")

            # Use legacy table format for validation comparison
            return self._build_legacy_comparison_table(comparison_data)

        except Exception as e:
            self.logger.exception("Failed to generate validation comparison")
            return html.P(f"Error generating comparison table: {str(e)}")

    def _build_legacy_comparison_table(self, comparison_data: List[Dict[str, Any]]) -> List[Any]:
        """Build legacy format comparison table (simpler version for validation state)"""
        # Separate numeric and categorical data
        numeric_data = [row for row in comparison_data if row.get('data_type') == 'numeric']
        categorical_data = [row for row in comparison_data if row.get('data_type') == 'categorical']

        content = []

        # Add summary info
        total_columns = len(comparison_data)
        content.append(html.P(f"Comparing {total_columns} columns", className="text-muted mb-3"))

        # Numeric columns table
        if numeric_data:
            content.append(html.H5("📊 Numeric Columns", className="mt-3"))
            numeric_table_data = []
            for row in numeric_data:
                numeric_table_data.append({
                    'table_column': f"{row['table']}.{row['column']}",
                    'type': f"{row['original_dtype']} → {row['synthetic_dtype']}",
                    'count': f"{row['original_count']:,} → {row['synthetic_count']:,}",
                    'missing': f"{row['original_missing']} → {row['synthetic_missing']}",
                    'mean': f"{row['original_mean']:.3f} → {row['synthetic_mean']:.3f}",
                    'median': f"{row['original_median']:.3f} → {row['synthetic_median']:.3f}",
                    'std': f"{row['original_std']:.3f} → {row['synthetic_std']:.3f}",
                    'iqr': f"{row['original_iqr']:.3f} → {row['synthetic_iqr']:.3f}",
                    'range': f"[{row['original_min']:.2f}, {row['original_max']:.2f}] → [{row['synthetic_min']:.2f}, {row['synthetic_max']:.2f}]"
                })

            content.append(
                self._build_table(
                    numeric_table_data,
                    ['table_column', 'type', 'count', 'missing', 'mean', 'median', 'std', 'iqr', 'range'],
                    {
                        'table_column': 'Table.Column',
                        'type': 'Data Type',
                        'count': 'Count (Orig → Synth)',
                        'missing': 'Missing',
                        'mean': 'Mean',
                        'median': 'Median',
                        'std': 'Std Dev',
                        'iqr': 'IQR',
                        'range': 'Range [min, max]'
                    }
                )
            )

        # Categorical columns table
        if categorical_data:
            content.append(html.H5("🏷️ Categorical Columns", className="mt-4"))
            categorical_table_data = []
            for row in categorical_data:
                categorical_table_data.append({
                    'table_column': f"{row['table']}.{row['column']}",
                    'type': f"{row['original_dtype']} → {row['synthetic_dtype']}",
                    'count': f"{row['original_count']:,} → {row['synthetic_count']:,}",
                    'missing': f"{row['original_missing']} → {row['synthetic_missing']}",
                    'unique': f"{row['original_unique']} → {row['synthetic_unique']}",
                    'mode': f"{row['original_mode']} → {row['synthetic_mode']}",
                    'mode_freq': f"{row['original_mode_freq']} ({row['original_mode_pct']:.1f}%) → {row['synthetic_mode_freq']} ({row['synthetic_mode_pct']:.1f}%)"
                })

            content.append(
                self._build_table(
                    categorical_table_data,
                    ['table_column', 'type', 'count', 'missing', 'unique', 'mode', 'mode_freq'],
                    {
                        'table_column': 'Table.Column',
                        'type': 'Data Type',
                        'count': 'Count (Orig → Synth)',
                        'missing': 'Missing',
                        'unique': 'Unique Values',
                        'mode': 'Most Frequent Value',
                        'mode_freq': 'Mode Frequency (%)'
                    }
                )
            )

        return content


def create_dashboard(host: str = '0.0.0.0', port: int = 8055) -> ValidationDashboard:
    """Create and return a validation dashboard instance"""
    return ValidationDashboard(host, port)


if __name__ == '__main__':
    dashboard = create_dashboard()
    dashboard.run(debug=True)
