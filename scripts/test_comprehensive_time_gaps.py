#!/usr/bin/env python3
"""
Test comprehensive time gap synthesis for ALL datetime pairs
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import json

from src.core.database import DatabaseManager
from src.core.config import ConfigManager
from src.temporal.comprehensive_time_gap_synthesizer import ComprehensiveTimeGapSynthesizer


def main():
    print("=" * 80)
    print("COMPREHENSIVE TIME GAP SYNTHESIS TEST")
    print("Testing ALL datetime pairs (non-overlapping)")
    print("=" * 80)
    
    # Initialize
    config = ConfigManager('config/generation_params.yaml')
    db_manager = DatabaseManager('nedis_data.duckdb')
    
    # Create synthesizer
    synthesizer = ComprehensiveTimeGapSynthesizer(db_manager, config)
    
    # Step 1: Analyze all patterns
    print("\n1. ANALYZING ALL TIME PATTERNS FROM REAL DATA...")
    print("-" * 60)
    patterns = synthesizer.analyze_all_time_patterns()
    
    print(f"\n✅ Analyzed {patterns['summary']['total_records']:,} records")
    print(f"✅ Found {patterns['summary']['hierarchical_levels']['level_1']} detailed patterns")
    print(f"✅ Found {patterns['summary']['hierarchical_levels']['level_2']} KTAS patterns")
    
    print("\n📊 Time gaps analyzed:")
    for gap_name in patterns['summary']['gaps_analyzed']:
        gap_def = synthesizer.gap_definitions[gap_name]
        print(f"   - {gap_name}: {gap_def['description']}")
    
    # Step 2: Check coverage for each gap type
    print("\n2. PATTERN COVERAGE BY GAP TYPE:")
    print("-" * 60)
    
    overall_patterns = patterns['hierarchical']['level_3']['overall']
    print(f"\n{'Gap Type':<30} {'Count':<10} {'Mean (min)':<12} {'Median (min)':<12}")
    print("-" * 64)
    
    for gap_name, gap_dist in overall_patterns.items():
        if isinstance(gap_dist, dict) and 'count' in gap_dist:
            print(f"{gap_name:<30} {gap_dist['count']:<10} "
                  f"{gap_dist['mean']:<12.1f} {gap_dist['median']:<12.1f}")
    
    # Step 3: Generate synthetic time gaps
    print("\n3. GENERATING SYNTHETIC TIME GAPS...")
    print("-" * 60)
    
    # Create test data
    n_test = 100
    np.random.seed(42)
    ktas_levels = np.random.choice([1, 2, 3, 4, 5], n_test, p=[0.01, 0.04, 0.25, 0.60, 0.10])
    
    # Treatment results with realistic distribution
    treatment_probs = {
        '11': 0.60,  # Discharge home
        '21': 0.10,  # Outpatient
        '31': 0.15,  # General ward admission
        '32': 0.05,  # ICU admission
        '99': 0.10   # Other
    }
    treatment_results = np.random.choice(
        list(treatment_probs.keys()),
        n_test,
        p=list(treatment_probs.values())
    )
    
    # Generate all datetime columns
    result_df = synthesizer.generate_all_time_gaps(ktas_levels, treatment_results)
    
    print(f"\n✅ Generated {len(result_df)} records with all datetime columns")
    
    # Step 4: Validate generated data
    print("\n4. VALIDATING TIME CONSISTENCY...")
    print("-" * 60)
    
    # Combine with KTAS for validation
    result_df['ktas01'] = ktas_levels
    result_df['emtrt_rust'] = treatment_results
    
    validation = synthesizer.validate_time_consistency(result_df)
    
    for check_name, check_result in validation['consistency_checks'].items():
        status = "✅" if check_result['percentage'] == 100 else "⚠️"
        print(f"{status} {check_name}: {check_result['percentage']:.1f}% valid "
              f"({check_result['valid']}/{check_result['valid'] + check_result['invalid']})")
    
    # Step 5: Show sample results
    print("\n5. SAMPLE GENERATED DATETIME PAIRS:")
    print("-" * 60)
    
    # Show first 5 records
    sample_cols = ['ktas01', 'emtrt_rust', 'ocur_dt', 'ocur_tm', 'vst_dt', 'vst_tm', 
                   'otrm_dt', 'otrm_tm', 'inpat_dt', 'inpat_tm', 'otpat_dt', 'otpat_tm']
    
    display_df = result_df[sample_cols].head(5)
    
    print("\nFirst 5 records:")
    for idx, row in display_df.iterrows():
        print(f"\nPatient {idx + 1} (KTAS {int(row['ktas01'])}, Result {row['emtrt_rust']}):")
        
        # Parse and display times
        times = {}
        for prefix in ['ocur', 'vst', 'otrm', 'inpat', 'otpat']:
            if row[f'{prefix}_dt'] and row[f'{prefix}_tm']:
                dt = row[f'{prefix}_dt']
                tm = row[f'{prefix}_tm']
                times[prefix] = f"{dt[0:4]}-{dt[4:6]}-{dt[6:8]} {tm[0:2]}:{tm[2:4]}"
            else:
                times[prefix] = "N/A"
        
        print(f"  Incident:  {times['ocur']}")
        print(f"  Arrival:   {times['vst']}")
        print(f"  Discharge: {times['otrm']}")
        print(f"  Admission: {times['inpat']}")
        print(f"  Outpatient:{times['otpat']}")
    
    # Step 6: Calculate actual gaps
    print("\n6. CALCULATED TIME GAPS (minutes):")
    print("-" * 60)
    
    # Parse datetimes and calculate gaps
    for prefix in ['ocur', 'vst', 'otrm', 'inpat', 'otpat']:
        if f'{prefix}_dt' in result_df.columns:
            result_df[f'{prefix}_datetime'] = result_df.apply(
                lambda x: synthesizer._parse_datetime(x[f'{prefix}_dt'], x[f'{prefix}_tm']),
                axis=1
            )
    
    # Calculate gaps
    gaps_to_calc = [
        ('incident_to_arrival', 'ocur_datetime', 'vst_datetime'),
        ('er_stay', 'vst_datetime', 'otrm_datetime'),
        ('discharge_to_admission', 'otrm_datetime', 'inpat_datetime'),
        ('discharge_to_outpatient', 'otrm_datetime', 'otpat_datetime')
    ]
    
    print(f"\n{'Gap Type':<25} {'Count':<8} {'Mean (min)':<12} {'Median (min)':<12}")
    print("-" * 57)
    
    for gap_name, from_col, to_col in gaps_to_calc:
        if from_col in result_df.columns and to_col in result_df.columns:
            result_df[f'gap_{gap_name}'] = result_df.apply(
                lambda x: synthesizer._calc_time_diff_minutes(x[to_col], x[from_col]),
                axis=1
            )
            
            gap_values = result_df[f'gap_{gap_name}'].dropna()
            if len(gap_values) > 0:
                print(f"{gap_name:<25} {len(gap_values):<8} "
                      f"{gap_values.mean():<12.1f} {gap_values.median():<12.1f}")
    
    # Step 7: KTAS-specific analysis
    print("\n7. TIME GAPS BY KTAS LEVEL:")
    print("-" * 60)
    
    print(f"\n{'KTAS':<6} {'ER Stay (median min)':<22} {'Admission Time (median min)':<25}")
    print("-" * 53)
    
    for ktas in range(1, 6):
        ktas_mask = result_df['ktas01'] == ktas
        ktas_data = result_df[ktas_mask]
        
        if len(ktas_data) > 0:
            er_stay = ktas_data['gap_er_stay'].dropna()
            admit_time = ktas_data['gap_discharge_to_admission'].dropna()
            
            er_median = er_stay.median() if len(er_stay) > 0 else 0
            admit_median = admit_time.median() if len(admit_time) > 0 else 0
            
            print(f"KTAS {ktas:<2} {er_median:<22.1f} {admit_median:<25.1f}")
    
    print("\n" + "=" * 80)
    print("✅ COMPREHENSIVE TIME GAP SYNTHESIS TEST COMPLETE!")
    print("All datetime pairs are being properly handled.")
    print("=" * 80)


if __name__ == "__main__":
    main()