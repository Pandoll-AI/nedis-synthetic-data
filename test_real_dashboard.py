#!/usr/bin/env python3
"""
Test script for the REAL data comparison dashboard.
Compares original nedis2017 (9.1M records) vs synthetic nedis2017 (100K records)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'validator'))

try:
    from validator.visualization.dashboard import create_dashboard

    def main():
        print("🚀 Starting NEDIS REAL Data Comparison Dashboard...")
        print("🔬 REAL Data Configuration:")
        print("   - Original DB: nedis_data.duckdb (9,123,382 records)")
        print("   - Synthetic DB: nedis_synth_2017.duckdb (100,502 synthetic records)")
        print("   - Table: nedis2017 with 87 columns")
        print("   - Includes: Demographics, vital signs, clinical data, KTAS scores")
        print("")
        print("🌐 Dashboard will be available at: http://localhost:8052")
        print("📝 Real Data Analysis Instructions:")
        print("   1. Click '🔄 Compare Tables' button")
        print("   2. Select 'nedis2017' from dropdown")
        print("   3. Choose comparison type:")
        print("      - 'All Columns': Compare all 87 columns")
        print("      - 'Numeric Only': Vital signs (BP, pulse, oxygen, etc.)")
        print("      - 'Categorical Only': Demographics, KTAS, treatment codes")
        print("   4. View REAL differences between original and synthetic data!")
        print("   5. Export results to CSV for further analysis")
        print("")

        dashboard = create_dashboard(host='localhost', port=8052)

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