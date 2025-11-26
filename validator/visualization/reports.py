"""
Report generation module for validation results.

This module provides:
- HTML report generation
- PDF report generation
- Excel report generation
- Interactive charts and visualizations
- Comprehensive validation summaries
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from ..core.validator import ValidationResult
from ..core.config import ValidationConfig


class ReportGenerator:
    """Advanced report generator for validation results"""

    def __init__(self, config: ValidationConfig):
        """
        Initialize report generator

        Args:
            config: Validation configuration
        """
        self.config = config
        self.output_dir = Path("reports")
        self.output_dir.mkdir(exist_ok=True)

    def generate_comprehensive_report(self, result: ValidationResult,
                                    formats: List[str] = None) -> str:
        """
        Generate comprehensive validation report

        Args:
            result: Validation result object
            formats: List of output formats

        Returns:
            Path to the main report file
        """
        if formats is None:
            formats = ['html']

        report_id = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        main_report_path = self.output_dir / f"{report_id}.html"

        # Generate HTML report
        if 'html' in formats:
            self._generate_html_report(result, main_report_path)

        # Generate PDF report
        if 'pdf' in formats:
            pdf_path = self.output_dir / f"{report_id}.pdf"
            self._generate_pdf_report(result, pdf_path)

        # Generate JSON report
        if 'json' in formats:
            json_path = self.output_dir / f"{report_id}.json"
            self._generate_json_report(result, json_path)

        # Generate Excel report
        if 'excel' in formats:
            excel_path = self.output_dir / f"{report_id}.xlsx"
            self._generate_excel_report(result, excel_path)

        return str(main_report_path)

    def _generate_html_report(self, result: ValidationResult, output_path: Path):
        """Generate HTML report with interactive charts"""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NEDIS Validation Report - {result.validation_type.title()}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 3px solid #007bff;
            padding-bottom: 20px;
        }}
        .score-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }}
        .score {{
            font-size: 4em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #495057;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }}
        .chart-container {{
            margin: 20px 0;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
        }}
        .error-list {{
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .warning-list {{
            background: #fff3cd;
            color: #856404;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🩺 NEDIS Synthetic Data Validation Report</h1>
            <h2>{result.validation_type.title()} Validation</h2>
            <p>Generated on: {result.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Duration: {result.duration:.2f} seconds</p>
        </div>

        <div class="score-card">
            <h3>Overall Validation Score</h3>
            <div class="score">{result.overall_score:.1f}/100</div>
        </div>

        <div class="section">
            <h2>📊 Validation Metrics</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <h4>Validation Type</h4>
                    <div class="metric-value">{result.validation_type.title()}</div>
                </div>
                <div class="metric-card">
                    <h4>Duration</h4>
                    <div class="metric-value">{result.duration:.2f}s</div>
                </div>
                <div class="metric-card">
                    <h4>Errors</h4>
                    <div class="metric-value">{len(result.errors)}</div>
                </div>
                <div class="metric-card">
                    <h4>Warnings</h4>
                    <div class="metric-value">{len(result.warnings)}</div>
                </div>
            </div>
        </div>
"""

        # Add category scores chart
        if result.results:
            html_content += """
        <div class="section">
            <h2>📈 Category Breakdown</h2>
            <div class="chart-container">
                <div id="categoryChart"></div>
            </div>
        </div>
"""

        # Add detailed results
        for category, category_result in result.results.items():
            if isinstance(category_result, dict) and 'overall_score' in category_result:
                html_content += f"""
        <div class="section">
            <h2>🔍 {category.title()} Analysis</h2>
            <div class="metric-card">
                <h4>Score</h4>
                <div class="metric-value">{category_result['overall_score']:.1f}/100</div>
            </div>
        </div>
"""

        # Add errors and warnings
        if result.errors:
            html_content += """
        <div class="section">
            <h2>❌ Errors</h2>
            <div class="error-list">
"""
            for error in result.errors:
                html_content += f"<p>• {error['error']}</p>"
            html_content += "</div></div>"

        if result.warnings:
            html_content += """
        <div class="section">
            <h2>⚠️ Warnings</h2>
            <div class="warning-list">
"""
            for warning in result.warnings:
                html_content += f"<p>• {warning['warning']}</p>"
            html_content += "</div></div>"

        # Add metadata
        if result.metadata:
            html_content += """
        <div class="section">
            <h2>📋 Metadata</h2>
            <div class="metric-grid">
"""
            for key, value in result.metadata.items():
                html_content += f"""
                <div class="metric-card">
                    <h4>{key.replace('_', ' ').title()}</h4>
                    <div class="metric-value">{value}</div>
                </div>
"""
            html_content += "</div></div>"

        # Add JavaScript for interactive charts
        html_content += """
        <script>
            // Category breakdown chart
            const categoryData = {
"""

        if result.results:
            category_names = []
            category_scores = []
            for category, category_result in result.results.items():
                if isinstance(category_result, dict) and 'overall_score' in category_result:
                    category_names.append(category.title())
                    category_scores.append(category_result['overall_score'])

            html_content += f"""
                x: {json.dumps(category_names)},
                y: {json.dumps(category_scores)},
                type: 'bar',
                marker: {{color: 'rgba(102, 126, 234, 0.8)'}}
            }}

            Plotly.newPlot('categoryChart', [categoryData], {{
                title: 'Category Scores',
                xaxis: {{title: 'Category'}},
                yaxis: {{title: 'Score'}}
            }});
            """

        html_content += """
        </script>
    </div>
</body>
</html>
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _generate_pdf_report(self, result: ValidationResult, output_path: Path):
        """Generate PDF report (placeholder)"""
        # This would require additional dependencies like reportlab
        # For now, create a simple text-based PDF placeholder
        pdf_content = f"""
NEDIS Validation Report
{result.validation_type.title()} Validation
Generated: {result.start_time.strftime('%Y-%m-%d %H:%M:%S')}
Overall Score: {result.overall_score:.1f}/100
Duration: {result.duration:.2f} seconds
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pdf_content)

    def _generate_json_report(self, result: ValidationResult, output_path: Path):
        """Generate JSON report"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    def _generate_excel_report(self, result: ValidationResult, output_path: Path):
        """Generate Excel report with multiple sheets"""
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = {
                    'Metric': ['Validation Type', 'Overall Score', 'Duration', 'Start Time', 'End Time', 'Errors', 'Warnings'],
                    'Value': [
                        result.validation_type,
                        result.overall_score,
                        result.duration,
                        result.start_time,
                        result.end_time,
                        len(result.errors),
                        len(result.warnings)
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

                # Results sheet
                if result.results:
                    results_data = []
                    for category, category_result in result.results.items():
                        if isinstance(category_result, dict) and 'overall_score' in category_result:
                            results_data.append({
                                'Category': category,
                                'Score': category_result['overall_score']
                            })

                    if results_data:
                        results_df = pd.DataFrame(results_data)
                        results_df.to_excel(writer, sheet_name='Results', index=False)

                # Errors sheet
                if result.errors:
                    errors_df = pd.DataFrame(result.errors)
                    errors_df.to_excel(writer, sheet_name='Errors', index=False)

                # Warnings sheet
                if result.warnings:
                    warnings_df = pd.DataFrame(result.warnings)
                    warnings_df.to_excel(writer, sheet_name='Warnings', index=False)

        except ImportError:
            # Fallback if openpyxl is not available
            self._generate_json_report(result, output_path.with_suffix('.json'))

    def generate_quick_report(self, result: ValidationResult) -> str:
        """
        Generate a quick summary report

        Args:
            result: Validation result object

        Returns:
            Summary report as string
        """
        report = []
        report.append("🩺 NEDIS Validation Quick Report")
        report.append("=" * 50)
        report.append(f"Validation Type: {result.validation_type.title()}")
        report.append(f"Overall Score: {result.overall_score:.1f}/100")
        report.append(f"Duration: {result.duration:.2f} seconds")
        report.append(f"Generated: {result.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        if result.results:
            report.append("\n📊 Category Scores:")
            for category, category_result in result.results.items():
                if isinstance(category_result, dict) and 'overall_score' in category_result:
                    report.append(f"  {category.title()}: {category_result['overall_score']:.1f}")

        if result.errors:
            report.append(f"\n❌ Errors: {len(result.errors)}")
            for i, error in enumerate(result.errors[:3], 1):
                report.append(f"  {i}. {error['error']}")

        if result.warnings:
            report.append(f"\n⚠️  Warnings: {len(result.warnings)}")
            for i, warning in enumerate(result.warnings[:3], 1):
                report.append(f"  {i}. {warning['warning']}")

        return "\n".join(report)


def create_report_generator(config: ValidationConfig) -> ReportGenerator:
    """Create and return a report generator instance"""
    return ReportGenerator(config)
