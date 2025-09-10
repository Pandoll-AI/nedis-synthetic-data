#!/usr/bin/env python3
"""
Time Gap Analysis and Synthesis Planning for NEDIS Data

This script analyzes time gaps between different events based on KTAS severity levels
and creates a comprehensive plan for time variable synthesis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_nedis_datetime(dt_str, tm_str):
    """
    Parse NEDIS date and time strings into datetime object
    
    Args:
        dt_str: Date string in format YYMMDD (e.g., '170123' for 2017-01-23)
        tm_str: Time string in format HHMM or HHMMSS (e.g., '0350' or '035000')
    
    Returns:
        datetime object or None if parsing fails
    """
    try:
        if pd.isna(dt_str) or pd.isna(tm_str) or dt_str == '' or tm_str == '':
            return None
            
        # Convert to string if needed
        dt_str = str(dt_str).strip()
        tm_str = str(tm_str).strip()
        
        # Parse date (YYMMDD -> 20YY-MM-DD)
        if len(dt_str) == 6:
            year = '20' + dt_str[0:2]
            month = dt_str[2:4]
            day = dt_str[4:6]
        else:
            return None
            
        # Parse time (HHMM or HHMMSS -> HH:MM:SS)
        if len(tm_str) == 4:
            hour = tm_str[0:2]
            minute = tm_str[2:4]
            second = '00'
        elif len(tm_str) == 6:
            hour = tm_str[0:2]
            minute = tm_str[2:4]
            second = tm_str[4:6]
        else:
            return None
            
        # Create datetime
        dt = datetime.strptime(f"{year}-{month}-{day} {hour}:{minute}:{second}", 
                               "%Y-%m-%d %H:%M:%S")
        return dt
    except:
        return None

def analyze_time_gaps(db_path='nedis_data.duckdb'):
    """
    Analyze time gaps between different events based on KTAS levels
    """
    conn = duckdb.connect(db_path)
    
    # Try different table names
    table_names = [
        'nedis_data.nedis2017',
        'nedis_original.nedis2017', 
        'main.nedis2017',
        'nedis2017'
    ]
    
    table_found = None
    for table_name in table_names:
        try:
            conn.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
            table_found = table_name
            logger.info(f"Found table: {table_name}")
            break
        except:
            continue
            
    if not table_found:
        logger.error("No NEDIS table found!")
        return None
        
    # Fetch data with datetime columns and KTAS levels
    query = f"""
    SELECT 
        ktas01,
        emtrt_rust,
        vst_dt, vst_tm,
        ocur_dt, ocur_tm,
        otrm_dt, otrm_tm,
        inpat_dt, inpat_tm,
        otpat_dt, otpat_tm
    FROM {table_found}
    WHERE ktas01 IS NOT NULL 
        AND ktas01 >= 1 
        AND ktas01 <= 5
    LIMIT 100000
    """
    
    logger.info("Fetching data...")
    df = conn.execute(query).fetchdf()
    logger.info(f"Fetched {len(df)} records")
    
    # Convert datetime columns
    logger.info("Parsing datetime columns...")
    df['vst_datetime'] = df.apply(lambda x: parse_nedis_datetime(x['vst_dt'], x['vst_tm']), axis=1)
    df['ocur_datetime'] = df.apply(lambda x: parse_nedis_datetime(x['ocur_dt'], x['ocur_tm']), axis=1)
    df['otrm_datetime'] = df.apply(lambda x: parse_nedis_datetime(x['otrm_dt'], x['otrm_tm']), axis=1)
    df['inpat_datetime'] = df.apply(lambda x: parse_nedis_datetime(x['inpat_dt'], x['inpat_tm']), axis=1)
    df['otpat_datetime'] = df.apply(lambda x: parse_nedis_datetime(x['otpat_dt'], x['otpat_tm']), axis=1)
    
    # Calculate time gaps
    logger.info("Calculating time gaps...")
    
    # Helper function to calculate time difference in minutes
    def calc_time_diff_minutes(dt1, dt2):
        if pd.notna(dt1) and pd.notna(dt2):
            if isinstance(dt1, datetime) and isinstance(dt2, datetime):
                return (dt1 - dt2).total_seconds() / 60
        return np.nan
    
    # 1. Occurrence to ER arrival (ocur -> vst)
    df['gap_ocur_to_vst'] = df.apply(lambda x: calc_time_diff_minutes(x['vst_datetime'], x['ocur_datetime']), axis=1)
    
    # 2. ER arrival to ER discharge (vst -> otrm)
    df['gap_vst_to_otrm'] = df.apply(lambda x: calc_time_diff_minutes(x['otrm_datetime'], x['vst_datetime']), axis=1)
    
    # 3. ER arrival to admission (vst -> inpat) for admitted patients
    df['gap_vst_to_inpat'] = df.apply(lambda x: calc_time_diff_minutes(x['inpat_datetime'], x['vst_datetime']), axis=1)
    
    # 4. ER arrival to outpatient (vst -> otpat) for outpatient referrals
    df['gap_vst_to_otpat'] = df.apply(lambda x: calc_time_diff_minutes(x['otpat_datetime'], x['vst_datetime']), axis=1)
    
    # Analyze distributions by KTAS level
    results = {}
    
    for ktas in range(1, 6):
        ktas_df = df[df['ktas01'] == ktas]
        
        results[f'ktas_{ktas}'] = {
            'total_count': len(ktas_df),
            'treatment_results': ktas_df['emtrt_rust'].value_counts().to_dict(),
            'time_gaps': {}
        }
        
        # Analyze each time gap
        gap_columns = ['gap_ocur_to_vst', 'gap_vst_to_otrm', 'gap_vst_to_inpat', 'gap_vst_to_otpat']
        
        for gap_col in gap_columns:
            valid_data = ktas_df[gap_col].dropna()
            valid_data = valid_data[valid_data > 0]  # Only positive gaps
            valid_data = valid_data[valid_data < 10080]  # Less than 1 week (reasonable limit)
            
            if len(valid_data) > 0:
                results[f'ktas_{ktas}']['time_gaps'][gap_col] = {
                    'count': len(valid_data),
                    'mean': float(valid_data.mean()),
                    'median': float(valid_data.median()),
                    'std': float(valid_data.std()),
                    'percentiles': {
                        '25': float(valid_data.quantile(0.25)),
                        '50': float(valid_data.quantile(0.50)),
                        '75': float(valid_data.quantile(0.75)),
                        '90': float(valid_data.quantile(0.90)),
                        '95': float(valid_data.quantile(0.95))
                    }
                }
    
    # Analyze by treatment result
    logger.info("Analyzing by treatment result...")
    
    treatment_results = df['emtrt_rust'].unique()
    for result in treatment_results:
        if pd.notna(result):
            result_df = df[df['emtrt_rust'] == result]
            
            for ktas in range(1, 6):
                ktas_result_df = result_df[result_df['ktas01'] == ktas]
                
                if len(ktas_result_df) > 10:  # Minimum sample size
                    key = f'ktas_{ktas}_result_{result}'
                    results[key] = {
                        'count': len(ktas_result_df),
                        'time_gaps': {}
                    }
                    
                    # Focus on relevant gaps based on treatment result
                    if result in ['1', '2', '3']:  # Admission codes
                        relevant_gaps = ['gap_vst_to_inpat', 'gap_vst_to_otrm']
                    elif result in ['4', '5']:  # Discharge codes  
                        relevant_gaps = ['gap_vst_to_otrm']
                    elif result == '6':  # Transfer
                        relevant_gaps = ['gap_vst_to_otrm']
                    else:
                        relevant_gaps = ['gap_vst_to_otrm']
                    
                    for gap_col in relevant_gaps:
                        valid_data = ktas_result_df[gap_col].dropna()
                        valid_data = valid_data[valid_data > 0]
                        valid_data = valid_data[valid_data < 10080]
                        
                        if len(valid_data) > 5:
                            results[key]['time_gaps'][gap_col] = {
                                'count': len(valid_data),
                                'mean': float(valid_data.mean()),
                                'median': float(valid_data.median()),
                                'std': float(valid_data.std())
                            }
    
    conn.close()
    return results

def create_synthesis_plan(analysis_results):
    """
    Create a comprehensive plan for time variable synthesis based on analysis
    """
    plan = {
        'overview': {
            'description': 'Time gap synthesis plan based on KTAS severity and treatment outcomes',
            'approach': 'Hierarchical probability-based generation with severity-adjusted distributions',
            'datetime_format': 'Convert YYMMDD + HHMM to proper datetime objects'
        },
        'hierarchical_strategy': {
            'level_1': 'KTAS level + Treatment result specific distributions',
            'level_2': 'KTAS level only distributions',
            'level_3': 'Overall average distributions',
            'fallback': 'Default reasonable values based on clinical guidelines'
        },
        'synthesis_rules': [],
        'implementation_steps': []
    }
    
    # Define synthesis rules based on KTAS levels
    ktas_rules = {
        '1': {
            'description': 'Critical/Resuscitation - Shortest time gaps',
            'typical_er_stay': '30-180 minutes',
            'admission_decision': '< 60 minutes',
            'priority': 'highest'
        },
        '2': {
            'description': 'Emergency - Very short time gaps',
            'typical_er_stay': '60-240 minutes',
            'admission_decision': '< 120 minutes',
            'priority': 'high'
        },
        '3': {
            'description': 'Urgent - Moderate time gaps',
            'typical_er_stay': '120-360 minutes',
            'admission_decision': '< 180 minutes',
            'priority': 'medium'
        },
        '4': {
            'description': 'Less urgent - Longer time gaps',
            'typical_er_stay': '180-480 minutes',
            'admission_decision': '< 240 minutes',
            'priority': 'low'
        },
        '5': {
            'description': 'Non-urgent - Longest time gaps',
            'typical_er_stay': '120-360 minutes',
            'admission_decision': 'Usually discharged',
            'priority': 'lowest'
        }
    }
    
    for ktas, rule in ktas_rules.items():
        if f'ktas_{ktas}' in analysis_results:
            stats = analysis_results[f'ktas_{ktas}']
            
            synthesis_rule = {
                'ktas_level': ktas,
                **rule,
                'observed_statistics': {
                    'sample_size': stats['total_count'],
                    'treatment_distribution': stats['treatment_results']
                },
                'time_gap_distributions': {}
            }
            
            # Add observed time gap distributions
            for gap_name, gap_stats in stats['time_gaps'].items():
                synthesis_rule['time_gap_distributions'][gap_name] = {
                    'method': 'log-normal distribution',
                    'parameters': {
                        'mean_minutes': gap_stats['mean'],
                        'std_minutes': gap_stats['std'],
                        'median_minutes': gap_stats['median']
                    },
                    'constraints': {
                        'min': gap_stats['percentiles']['25'] * 0.5,
                        'max': gap_stats['percentiles']['95'] * 1.5
                    }
                }
            
            plan['synthesis_rules'].append(synthesis_rule)
    
    # Implementation steps
    plan['implementation_steps'] = [
        {
            'step': 1,
            'name': 'Create TimeGapSynthesizer class',
            'description': 'Core class for time gap generation based on KTAS and treatment outcome',
            'location': 'src/temporal/time_gap_synthesizer.py'
        },
        {
            'step': 2,
            'name': 'Build distribution models',
            'description': 'Create log-normal distribution models for each KTAS-treatment combination',
            'method': 'Use scipy.stats.lognorm with observed parameters'
        },
        {
            'step': 3,
            'name': 'Implement hierarchical fallback',
            'description': 'Fallback from specific to general distributions when data is sparse',
            'levels': ['KTAS+Treatment', 'KTAS only', 'Overall average']
        },
        {
            'step': 4,
            'name': 'Add clinical constraints',
            'description': 'Ensure generated times follow clinical logic',
            'constraints': [
                'ocur_datetime <= vst_datetime',
                'vst_datetime <= otrm_datetime',
                'vst_datetime <= inpat_datetime (if admitted)',
                'Minimum gap thresholds based on KTAS'
            ]
        },
        {
            'step': 5,
            'name': 'Integrate with VectorizedPatientGenerator',
            'description': 'Add time gap synthesis to main generation pipeline',
            'integration_point': 'After KTAS and treatment result assignment'
        }
    ]
    
    return plan

def main():
    """Main execution"""
    logger.info("Starting time gap analysis and synthesis planning...")
    
    # Analyze time gaps
    analysis_results = analyze_time_gaps()
    
    if analysis_results:
        # Save analysis results
        output_dir = Path('outputs')
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / 'time_gap_analysis.json', 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        logger.info(f"Saved analysis results to outputs/time_gap_analysis.json")
        
        # Create synthesis plan
        synthesis_plan = create_synthesis_plan(analysis_results)
        
        with open(output_dir / 'time_gap_synthesis_plan.json', 'w') as f:
            json.dump(synthesis_plan, f, indent=2)
        logger.info(f"Saved synthesis plan to outputs/time_gap_synthesis_plan.json")
        
        # Print summary
        print("\n" + "="*60)
        print("TIME GAP ANALYSIS SUMMARY")
        print("="*60)
        
        for ktas in range(1, 6):
            if f'ktas_{ktas}' in analysis_results:
                stats = analysis_results[f'ktas_{ktas}']
                print(f"\nKTAS Level {ktas}:")
                print(f"  Sample size: {stats['total_count']}")
                
                if 'gap_vst_to_otrm' in stats['time_gaps']:
                    gap_stats = stats['time_gaps']['gap_vst_to_otrm']
                    print(f"  ER Stay (vst->otrm):")
                    print(f"    Mean: {gap_stats['mean']:.1f} min")
                    print(f"    Median: {gap_stats['median']:.1f} min")
                    print(f"    Std: {gap_stats['std']:.1f} min")
                
                if 'gap_vst_to_inpat' in stats['time_gaps']:
                    gap_stats = stats['time_gaps']['gap_vst_to_inpat']
                    print(f"  Time to Admission (vst->inpat):")
                    print(f"    Mean: {gap_stats['mean']:.1f} min")
                    print(f"    Median: {gap_stats['median']:.1f} min")
        
        print("\n" + "="*60)
        print("SYNTHESIS PLAN CREATED")
        print("="*60)
        print("\nNext steps:")
        for step in synthesis_plan['implementation_steps']:
            print(f"{step['step']}. {step['name']}")
            print(f"   {step['description']}")
    else:
        logger.error("Analysis failed - no results generated")

if __name__ == "__main__":
    main()