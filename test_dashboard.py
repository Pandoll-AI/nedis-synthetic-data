#!/usr/bin/env python3
"""
Test script for the enhanced dashboard with table comparison features.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'validator'))

try:
    from validator.visualization.dashboard import create_dashboard

    def main():
        print("🚀 Starting NEDIS Table Comparison Dashboard...")
        print("📊 Features included:")
        print("   - Immediate table comparison without validation")
        print("   - Interactive table/column selection")
        print("   - Enhanced statistics with color coding")
        print("   - Visual difference highlighting")
        print("   - Distribution comparison charts")
        print("   - CSV export functionality")
        print("")
        print("🌐 Dashboard will be available at: http://localhost:8050")
        print("📝 Make sure you have sample databases available for testing")
        print("")

        dashboard = create_dashboard(host='localhost', port=8050)

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