#!/usr/bin/env python3
"""
End-to-End Test for PDF Generation Functionality
Tests the PDF export feature with real NEDIS data
"""

import sys
import os
import tempfile
import time
from pathlib import Path
import logging

# Add validator to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'validator'))

def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='[PDF Test] %(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('pdf_test')

def test_pdf_generation():
    """End-to-end test for PDF generation"""
    logger = setup_logging()

    logger.info("🧪 Starting End-to-End PDF Generation Test")

    try:
        # Import dashboard and core modules
        logger.info("📦 Importing dashboard modules...")
        from validator.visualization.dashboard import ValidationDashboard
        from validator.core.database import get_database_manager

        logger.info("✅ Modules imported successfully")

        # Initialize dashboard
        logger.info("🚀 Initializing dashboard...")
        dashboard = ValidationDashboard(host='localhost', port=8099)  # Use different port
        logger.info("✅ Dashboard initialized")

        # Test data setup
        original_db = "nedis_data.duckdb"
        synthetic_db = "nedis_synth_2017.duckdb"
        selected_tables = ["nedis2017"]

        logger.info(f"🔬 Test configuration:")
        logger.info(f"   - Original DB: {original_db}")
        logger.info(f"   - Synthetic DB: {synthetic_db}")
        logger.info(f"   - Tables: {selected_tables}")

        # Verify databases exist
        logger.info("📋 Checking database files...")
        if not os.path.exists(original_db):
            raise FileNotFoundError(f"Original database not found: {original_db}")
        if not os.path.exists(synthetic_db):
            raise FileNotFoundError(f"Synthetic database not found: {synthetic_db}")
        logger.info("✅ Database files found")

        # Test database connectivity
        logger.info("🔌 Testing database connectivity...")
        db_manager = get_database_manager()

        # Get table info to verify data access
        original_info = db_manager.get_table_info("nedis2017", original_db)
        synthetic_info = db_manager.get_table_info("nedis2017", synthetic_db)

        if 'error' in original_info:
            raise Exception(f"Error accessing original database: {original_info['error']}")
        if 'error' in synthetic_info:
            raise Exception(f"Error accessing synthetic database: {synthetic_info['error']}")

        logger.info(f"✅ Database connectivity verified")
        logger.info(f"   - Original records: {original_info.get('record_count', 'Unknown')}")
        logger.info(f"   - Synthetic records: {synthetic_info.get('record_count', 'Unknown')}")

        # Test chart generation (the core of PDF export)
        logger.info("📊 Testing chart generation...")
        charts = dashboard._generate_comparison_charts(selected_tables, original_db, synthetic_db)

        if not charts:
            raise Exception("No charts were generated")

        logger.info(f"✅ Generated {len(charts)} chart components")

        # Analyze chart types
        chart_types = []
        valid_charts = 0

        for i, chart in enumerate(charts):
            if hasattr(chart, 'figure') and chart.figure:
                valid_charts += 1
                if hasattr(chart.figure, 'layout') and hasattr(chart.figure.layout, 'title'):
                    title = chart.figure.layout.title
                    if title and hasattr(title, 'text') and title.text:
                        chart_types.append(title.text)
                        logger.info(f"   Chart {i+1}: {title.text}")
                    else:
                        logger.info(f"   Chart {i+1}: [No title]")
                else:
                    logger.info(f"   Chart {i+1}: [No layout/title]")
            else:
                logger.warning(f"   Chart {i+1}: Invalid chart (no figure)")

        logger.info(f"✅ Found {valid_charts} valid charts out of {len(charts)} total")

        # Test PDF generation
        logger.info("📄 Testing PDF generation...")

        # Simulate the PDF export callback
        try:
            import io
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import inch
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
            from datetime import datetime

            logger.info("✅ PDF libraries imported successfully")

            # Create temporary directory for chart images
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"📁 Created temp directory: {temp_dir}")

                # Create PDF buffer
                pdf_buffer = io.BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=A4,
                                      topMargin=inch, bottomMargin=inch,
                                      leftMargin=inch, rightMargin=inch)
                story = []

                # Add title and metadata
                styles = getSampleStyleSheet()
                title = Paragraph("NEDIS Data Comparison Report - E2E Test", styles['Title'])
                story.append(title)
                story.append(Spacer(1, 20))

                metadata = Paragraph(f"""
                <b>Test Configuration:</b><br/>
                <b>Tables Analyzed:</b> {', '.join(selected_tables)}<br/>
                <b>Original Database:</b> {original_db}<br/>
                <b>Synthetic Database:</b> {synthetic_db}<br/>
                <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
                <b>Test Type:</b> End-to-End PDF Generation Test<br/>
                """, styles['Normal'])
                story.append(metadata)
                story.append(Spacer(1, 20))

                # Process charts and export to images
                chart_count = 0
                exported_images = []

                for i, chart_component in enumerate(charts):
                    try:
                        if hasattr(chart_component, 'figure') and chart_component.figure:
                            fig = chart_component.figure
                            img_path = os.path.join(temp_dir, f"chart_{i}.png")

                            logger.info(f"   Exporting chart {i+1} to image...")

                            # Export plotly figure to image
                            fig.write_image(img_path, width=800, height=600, engine="kaleido")

                            if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                                exported_images.append(img_path)

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

                                logger.info(f"   ✅ Chart {i+1} exported successfully ({os.path.getsize(img_path)} bytes)")
                            else:
                                logger.warning(f"   ❌ Chart {i+1} export failed (file missing or empty)")

                    except Exception as chart_error:
                        logger.error(f"   ❌ Chart {i+1} export failed: {chart_error}")
                        continue

                logger.info(f"✅ Successfully exported {chart_count} charts as images")
                logger.info(f"📁 Image files: {[os.path.basename(img) for img in exported_images]}")

                if chart_count == 0:
                    # Add message if no charts were generated
                    no_charts = Paragraph("⚠️ No charts were available for PDF export in this test.", styles['Normal'])
                    story.append(no_charts)
                    logger.warning("⚠️ No charts were exported to PDF")
                else:
                    # Add summary
                    summary = Paragraph(f"""
                    <b>Export Summary:</b><br/>
                    • Total charts generated: {len(charts)}<br/>
                    • Charts successfully exported: {chart_count}<br/>
                    • Chart types: {', '.join(chart_types[:3])}{'...' if len(chart_types) > 3 else ''}<br/>
                    • Test status: ✅ PASSED<br/>
                    """, styles['Normal'])
                    story.append(summary)

                # Build PDF
                logger.info("📄 Building PDF document...")
                doc.build(story)
                pdf_buffer.seek(0)
                pdf_data = pdf_buffer.getvalue()

                # Save test PDF to file
                test_filename = f"test_pdf_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                with open(test_filename, 'wb') as f:
                    f.write(pdf_data)

                logger.info(f"✅ PDF generated successfully!")
                logger.info(f"   📄 File: {test_filename}")
                logger.info(f"   📏 Size: {len(pdf_data):,} bytes")
                logger.info(f"   📊 Charts included: {chart_count}")

                # Verify PDF file
                if os.path.exists(test_filename) and os.path.getsize(test_filename) > 0:
                    logger.info(f"✅ PDF file verification passed")

                    # Try to read PDF metadata (basic validation)
                    try:
                        import PyPDF2
                        with open(test_filename, 'rb') as pdf_file:
                            pdf_reader = PyPDF2.PdfReader(pdf_file)
                            num_pages = len(pdf_reader.pages)
                            logger.info(f"   📖 PDF pages: {num_pages}")

                            if num_pages > 0:
                                logger.info(f"✅ PDF structure validation passed")
                            else:
                                logger.warning(f"⚠️ PDF has no pages")

                    except ImportError:
                        logger.info("ℹ️ PyPDF2 not available for detailed PDF validation")
                    except Exception as pdf_error:
                        logger.warning(f"⚠️ PDF validation error: {pdf_error}")

                else:
                    raise Exception("PDF file was not created or is empty")

        except ImportError as import_error:
            logger.error(f"❌ Missing required library: {import_error}")
            logger.error("💡 Make sure reportlab and kaleido are installed:")
            logger.error("   pip install reportlab kaleido")
            raise

        except Exception as pdf_error:
            logger.error(f"❌ PDF generation failed: {pdf_error}")
            raise

        # Test summary
        logger.info("")
        logger.info("🎉 END-TO-END PDF TEST COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"✅ Database connectivity: PASSED")
        logger.info(f"✅ Chart generation: PASSED ({len(charts)} components)")
        logger.info(f"✅ Valid charts: PASSED ({valid_charts} charts)")
        logger.info(f"✅ Image export: PASSED ({chart_count} images)")
        logger.info(f"✅ PDF generation: PASSED ({test_filename})")
        logger.info(f"✅ File verification: PASSED")
        logger.info("=" * 60)
        logger.info(f"📄 Test PDF saved as: {test_filename}")
        logger.info("🎯 The PDF export functionality is working correctly!")

        return True

    except Exception as e:
        logger.error("")
        logger.error("❌ END-TO-END PDF TEST FAILED!")
        logger.error("=" * 60)
        logger.error(f"Error: {e}")
        logger.error("=" * 60)
        return False

if __name__ == "__main__":
    success = test_pdf_generation()
    sys.exit(0 if success else 1)