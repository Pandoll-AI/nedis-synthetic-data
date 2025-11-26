#!/usr/bin/env python3
"""
Demo script for the enhanced dashboard - uses same database for comparison demo.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'validator'))

try:
    from validator.visualization.dashboard import ValidationDashboard

    def main():
        print("🚀 Starting NEDIS Table Comparison Dashboard (Demo Mode)...")
        print("📊 Demo Configuration:")
        print("   - Original DB: nedis_data.duckdb")
        print("   - For demo: Using same DB for comparison")
        print("   - Available tables: diag_adm, diag_er, iciss, nedis2017")
        print("   - Records in nedis2017: 9,123,382")
        print("")
        print("🌐 Dashboard will be available at: http://localhost:8051")
        print("📝 Instructions:")
        print("   1. Click '🔄 Compare Tables' button")
        print("   2. Select one or more tables from dropdown")
        print("   3. Choose comparison type (All/Numeric/Categorical)")
        print("   4. View enhanced statistics and charts")
        print("")

        # Create dashboard with demo port
        dashboard = ValidationDashboard(host='localhost', port=8051)

        try:
            dashboard.run(debug=True)
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped by user")
        except Exception as e:
            print(f"❌ Error running dashboard: {e}")

    if __name__ == '__main__':
        main()

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📋 Make sure all dependencies are installed:")
    print("   pip install dash dash-bootstrap-components plotly pandas numpy scipy")