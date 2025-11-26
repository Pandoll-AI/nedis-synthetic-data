#!/usr/bin/env python3
"""
Generate REAL synthetic data by sampling and modifying original nedis2017 data.
This creates realistic synthetic data with controlled variations.
"""

import duckdb
import pandas as pd
import numpy as np

def create_synthetic_data():
    print("🔬 Connecting to original database...")

    # Connect to original database
    original_conn = duckdb.connect('nedis_data.duckdb', read_only=True)

    print("📊 Sampling data from nedis2017 table...")

    # Sample about 100,000 records (1.1% of 9M) with random sampling
    query = """
    SELECT *
    FROM nedis2017
    WHERE random() < 0.011
    ORDER BY random()
    """

    # Get sample data
    df = original_conn.execute(query).fetchdf()
    print(f"✅ Sampled {len(df):,} records")

    if len(df) == 0:
        print("❌ No data sampled. Trying with higher sampling rate...")
        query = """
        SELECT *
        FROM nedis2017
        LIMIT 50000
        """
        df = original_conn.execute(query).fetchdf()
        print(f"✅ Got {len(df):,} records via LIMIT")

    original_conn.close()

    print("🧬 Generating synthetic variations...")

    # Create synthetic variations
    synthetic_df = df.copy()

    # Modify identifiers
    synthetic_df['index_key'] = ['SYNTH_' + str(i).zfill(8) for i in range(len(synthetic_df))]
    synthetic_df['pat_reg_no'] = ['PAT_' + str(np.random.randint(100000, 999999)) for _ in range(len(synthetic_df))]

    # Add realistic noise to vital signs (±5% variation)
    vital_cols = ['vst_sbp', 'vst_dbp', 'vst_per_pu', 'vst_per_br', 'vst_oxy']

    for col in vital_cols:
        if col in synthetic_df.columns:
            # Only modify positive values, keep -1 (missing) as is
            mask = synthetic_df[col] > 0
            if mask.any():
                noise = np.random.uniform(0.95, 1.05, mask.sum())
                synthetic_df.loc[mask, col] = (synthetic_df.loc[mask, col] * noise).astype(int)

    # Anonymize patient names if present
    if 'pat_nm' in synthetic_df.columns:
        synthetic_df['pat_nm'] = ['SYNTH_NAME_' + str(i).zfill(6) for i in range(len(synthetic_df))]

    print("💾 Saving synthetic data...")

    # Connect to synthetic database and save
    synthetic_conn = duckdb.connect('nedis_synth_2017.duckdb')

    # Drop test table if exists
    synthetic_conn.execute("DROP TABLE IF EXISTS test")

    # Create nedis2017 table in synthetic database
    synthetic_conn.execute("DROP TABLE IF EXISTS nedis2017")
    synthetic_conn.register('synthetic_data', synthetic_df)
    synthetic_conn.execute("CREATE TABLE nedis2017 AS SELECT * FROM synthetic_data")

    # Verify the creation
    count = synthetic_conn.execute("SELECT COUNT(*) FROM nedis2017").fetchone()[0]
    print(f"✅ Created nedis2017 table with {count:,} synthetic records")

    # Show sample of synthetic data
    sample = synthetic_conn.execute("SELECT pat_age_gr, pat_sex, ktas_fstu, vst_sbp, vst_dbp, vst_per_pu, emtrt_rust FROM nedis2017 LIMIT 5").fetchall()
    print("\n📋 Sample synthetic data:")
    for row in sample:
        print(f"   {row}")

    synthetic_conn.close()

    print("\n🎉 Real synthetic data creation completed!")
    print("📍 You can now use:")
    print("   Original DB: nedis_data.duckdb")
    print("   Synthetic DB: nedis_synth_2017.duckdb")
    print("   Both contain nedis2017 table for real comparison!")

if __name__ == "__main__":
    create_synthetic_data()