#!/usr/bin/env python3
"""
Test Time Gap Synthesis with Real NEDIS Data

This script tests the TimeGapSynthesizer implementation using real data from nedis_data.duckdb
and validates that the generated time gaps follow the expected distributions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

from src.core.database import DatabaseManager
from src.core.config import ConfigManager
from src.temporal.time_gap_synthesizer import TimeGapSynthesizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_original_data(db_path='nedis_data.duckdb', sample_size=10000):
    """Load original NEDIS data for comparison"""
    logger.info(f"Loading original data from {db_path}...")
    
    db = DatabaseManager(db_path)
    
    query = f"""
    SELECT 
        ktas01,
        emtrt_rust,
        vst_dt, vst_tm,
        otrm_dt, otrm_tm,
        inpat_dt, inpat_tm
    FROM nedis2017
    WHERE ktas01 IS NOT NULL 
        AND ktas01 >= 1 
        AND ktas01 <= 5
        AND vst_dt IS NOT NULL
        AND vst_tm IS NOT NULL
    LIMIT {sample_size}
    """
    
    df = db.fetch_dataframe(query)
    logger.info(f"Loaded {len(df)} original records")
    
    return df

def parse_datetime(dt_str, tm_str):
    """Parse NEDIS datetime format (supports both YYYYMMDD and YYMMDD)"""
    try:
        if pd.isna(dt_str) or pd.isna(tm_str):
            return None
            
        dt_str = str(dt_str).strip()
        tm_str = str(tm_str).strip().zfill(4)
        
        # Handle both YYYYMMDD (8 digits) and YYMMDD (6 digits) formats
        if len(dt_str) == 8:
            year = dt_str[0:4]
            month = dt_str[4:6]
            day = dt_str[6:8]
        elif len(dt_str) == 6:
            year = '20' + dt_str[0:2]
            month = dt_str[2:4]
            day = dt_str[4:6]
        else:
            return None
            
        hour = tm_str[0:2]
        minute = tm_str[2:4]
        
        return datetime.strptime(
            f"{year}-{month}-{day} {hour}:{minute}",
            "%Y-%m-%d %H:%M"
        )
    except:
        pass
    return None

def calculate_original_time_gaps(df):
    """Calculate time gaps from original data"""
    logger.info("Calculating original time gaps...")
    
    # Parse datetimes
    df['vst_datetime'] = df.apply(lambda x: parse_datetime(x['vst_dt'], x['vst_tm']), axis=1)
    df['otrm_datetime'] = df.apply(lambda x: parse_datetime(x['otrm_dt'], x['otrm_tm']), axis=1)
    df['inpat_datetime'] = df.apply(lambda x: parse_datetime(x['inpat_dt'], x['inpat_tm']), axis=1)
    
    # Calculate gaps using apply to handle None values
    def calc_time_diff(dt1, dt2):
        if pd.notna(dt1) and pd.notna(dt2) and isinstance(dt1, datetime) and isinstance(dt2, datetime):
            return (dt1 - dt2).total_seconds() / 60
        return np.nan
    
    df['original_er_stay'] = df.apply(lambda x: calc_time_diff(x['otrm_datetime'], x['vst_datetime']), axis=1)
    df['original_admit_time'] = df.apply(lambda x: calc_time_diff(x['inpat_datetime'], x['vst_datetime']), axis=1)
    
    # Filter reasonable values
    df.loc[df['original_er_stay'] <= 0, 'original_er_stay'] = np.nan
    df.loc[df['original_er_stay'] > 10080, 'original_er_stay'] = np.nan  # > 1 week
    df.loc[df['original_admit_time'] <= 0, 'original_admit_time'] = np.nan
    df.loc[df['original_admit_time'] > 10080, 'original_admit_time'] = np.nan
    
    return df

def test_time_gap_generation(original_df):
    """Test time gap generation using the synthesizer"""
    logger.info("Testing time gap synthesis...")
    
    # Initialize synthesizer with real data
    db = DatabaseManager('nedis_data.duckdb')
    config = ConfigManager()
    
    # Override source table in config
    config.config = {'original': {'source_table': 'nedis2017'}}
    
    synthesizer = TimeGapSynthesizer(db, config)
    
    # Analyze patterns from real data
    patterns = synthesizer.analyze_time_patterns()
    logger.info(f"Analyzed {len(patterns)} pattern groups")
    
    # Generate synthetic time gaps
    synthetic_gaps = synthesizer.generate_time_gaps(
        ktas_levels=original_df['ktas01'].values,
        treatment_results=original_df['emtrt_rust'].values,
        visit_datetimes=original_df['vst_datetime']
    )
    
    # Parse synthetic datetimes - need to align indices
    original_df = original_df.reset_index(drop=True)
    synthetic_gaps = synthetic_gaps.reset_index(drop=True)
    
    # Calculate synthetic time gaps
    def calc_gap(row_idx):
        if row_idx < len(synthetic_gaps) and row_idx < len(original_df):
            synth_otrm = synthetic_gaps.loc[row_idx, 'otrm_datetime']
            orig_vst = original_df.loc[row_idx, 'vst_datetime']
            if pd.notna(synth_otrm) and pd.notna(orig_vst) and isinstance(synth_otrm, datetime) and isinstance(orig_vst, datetime):
                return (synth_otrm - orig_vst).total_seconds() / 60
        return np.nan
    
    synthetic_gaps['synthetic_er_stay'] = [calc_gap(i) for i in range(len(synthetic_gaps))]
    
    # Calculate admission time gaps
    def calc_admit_gap(row_idx):
        if row_idx < len(synthetic_gaps) and row_idx < len(original_df):
            synth_inpat = synthetic_gaps.loc[row_idx, 'inpat_datetime']
            orig_vst = original_df.loc[row_idx, 'vst_datetime']
            if pd.notna(synth_inpat) and pd.notna(orig_vst) and isinstance(synth_inpat, datetime) and isinstance(orig_vst, datetime):
                return (synth_inpat - orig_vst).total_seconds() / 60
        return np.nan
    
    synthetic_gaps['synthetic_admit_time'] = [calc_admit_gap(i) for i in range(len(synthetic_gaps))]
    
    return synthetic_gaps, patterns

def compare_distributions(original_df, synthetic_gaps):
    """Compare original and synthetic distributions"""
    logger.info("Comparing distributions...")
    
    results = {}
    
    # Compare by KTAS level
    for ktas in range(1, 6):
        ktas_mask = original_df['ktas01'] == ktas
        
        # ER stay comparison
        orig_er_stay = original_df.loc[ktas_mask, 'original_er_stay'].dropna()
        synth_er_stay = synthetic_gaps.loc[ktas_mask, 'synthetic_er_stay'].dropna()
        
        if len(orig_er_stay) > 10 and len(synth_er_stay) > 10:
            # KS test
            ks_stat, ks_pval = stats.ks_2samp(orig_er_stay, synth_er_stay)
            
            results[f'ktas_{ktas}_er_stay'] = {
                'original_mean': float(orig_er_stay.mean()),
                'original_median': float(orig_er_stay.median()),
                'original_std': float(orig_er_stay.std()),
                'synthetic_mean': float(synth_er_stay.mean()),
                'synthetic_median': float(synth_er_stay.median()),
                'synthetic_std': float(synth_er_stay.std()),
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pval),
                'distribution_similar': bool(ks_pval > 0.05)
            }
        
        # Admission time comparison (for admitted patients)
        admit_mask = ktas_mask & original_df['emtrt_rust'].isin(['31', '32', '33', '34'])
        orig_admit = original_df.loc[admit_mask, 'original_admit_time'].dropna()
        synth_admit = synthetic_gaps.loc[admit_mask, 'synthetic_admit_time'].dropna()
        
        if len(orig_admit) > 10 and len(synth_admit) > 10:
            ks_stat, ks_pval = stats.ks_2samp(orig_admit, synth_admit)
            
            results[f'ktas_{ktas}_admit_time'] = {
                'original_mean': float(orig_admit.mean()),
                'synthetic_mean': float(synth_admit.mean()),
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pval),
                'distribution_similar': bool(ks_pval > 0.05)
            }
    
    return results

def visualize_comparisons(original_df, synthetic_gaps, output_dir='outputs'):
    """Create visualization comparing distributions"""
    logger.info("Creating visualizations...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create figure with subplots for each KTAS level
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle('Time Gap Distributions: Original vs Synthetic by KTAS Level', fontsize=16)
    
    for i, ktas in enumerate(range(1, 6)):
        ktas_mask = original_df['ktas01'] == ktas
        
        # ER Stay distributions
        ax = axes[0, i]
        orig_er = original_df.loc[ktas_mask, 'original_er_stay'].dropna()
        synth_er = synthetic_gaps.loc[ktas_mask, 'synthetic_er_stay'].dropna()
        
        if len(orig_er) > 0:
            ax.hist(orig_er, bins=30, alpha=0.5, label='Original', color='blue', density=True)
        if len(synth_er) > 0:
            ax.hist(synth_er, bins=30, alpha=0.5, label='Synthetic', color='red', density=True)
        
        ax.set_title(f'KTAS {ktas}: ER Stay')
        ax.set_xlabel('Minutes')
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_xlim(0, min(1000, max(orig_er.max() if len(orig_er) > 0 else 0, 
                                     synth_er.max() if len(synth_er) > 0 else 0)))
        
        # Q-Q plots
        ax = axes[1, i]
        if len(orig_er) > 0 and len(synth_er) > 0:
            # Sample same size for Q-Q plot
            sample_size = min(len(orig_er), len(synth_er))
            orig_sample = np.random.choice(orig_er, sample_size, replace=False)
            synth_sample = np.random.choice(synth_er, sample_size, replace=False)
            
            stats.probplot(orig_sample, dist="norm", plot=ax)
            ax.get_lines()[0].set_markerfacecolor('blue')
            ax.get_lines()[0].set_alpha(0.5)
            stats.probplot(synth_sample, dist="norm", plot=ax)
            ax.get_lines()[2].set_markerfacecolor('red')
            ax.get_lines()[2].set_alpha(0.5)
        
        ax.set_title(f'KTAS {ktas}: Q-Q Plot')
        
        # Box plots
        ax = axes[2, i]
        data_to_plot = []
        labels = []
        
        if len(orig_er) > 0:
            data_to_plot.append(orig_er[orig_er < 1000])  # Limit for visualization
            labels.append('Original')
        if len(synth_er) > 0:
            data_to_plot.append(synth_er[synth_er < 1000])
            labels.append('Synthetic')
        
        if data_to_plot:
            ax.boxplot(data_to_plot, labels=labels)
        ax.set_title(f'KTAS {ktas}: Box Plot')
        ax.set_ylabel('Minutes')
    
    plt.tight_layout()
    plt.savefig(output_path / 'time_gap_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved visualization to {output_path / 'time_gap_comparison.png'}")
    
    # Close figure to free memory
    plt.close()

def print_summary(comparison_results):
    """Print summary of comparison results"""
    print("\n" + "="*80)
    print("TIME GAP SYNTHESIS TEST RESULTS")
    print("="*80)
    
    # ER Stay results
    print("\n### ER Stay Time Comparisons ###")
    print(f"{'KTAS':<6} {'Orig Mean':<12} {'Synth Mean':<12} {'Difference':<12} {'KS p-value':<12} {'Similar?':<10}")
    print("-"*70)
    
    for ktas in range(1, 6):
        key = f'ktas_{ktas}_er_stay'
        if key in comparison_results:
            res = comparison_results[key]
            diff_pct = ((res['synthetic_mean'] - res['original_mean']) / res['original_mean'] * 100)
            print(f"{ktas:<6} {res['original_mean']:<12.1f} {res['synthetic_mean']:<12.1f} "
                  f"{diff_pct:<12.1f}% {res['ks_pvalue']:<12.4f} {'✓' if res['distribution_similar'] else '✗':<10}")
    
    # Admission time results
    print("\n### Admission Time Comparisons ###")
    print(f"{'KTAS':<6} {'Orig Mean':<12} {'Synth Mean':<12} {'KS p-value':<12} {'Similar?':<10}")
    print("-"*60)
    
    for ktas in range(1, 6):
        key = f'ktas_{ktas}_admit_time'
        if key in comparison_results:
            res = comparison_results[key]
            print(f"{ktas:<6} {res['original_mean']:<12.1f} {res['synthetic_mean']:<12.1f} "
                  f"{res['ks_pvalue']:<12.4f} {'✓' if res['distribution_similar'] else '✗':<10}")
    
    # Overall assessment
    similar_count = sum(1 for r in comparison_results.values() if r.get('distribution_similar', False))
    total_count = len(comparison_results)
    
    print("\n" + "="*80)
    print(f"OVERALL: {similar_count}/{total_count} distributions are statistically similar (p > 0.05)")
    print("="*80)

def main():
    """Main test execution"""
    logger.info("Starting time gap synthesis test with real data...")
    
    # Load original data
    original_df = load_original_data(sample_size=10000)
    
    # Calculate original time gaps
    original_df = calculate_original_time_gaps(original_df)
    
    # Test synthesis
    synthetic_gaps, patterns = test_time_gap_generation(original_df)
    
    # Merge results for comparison
    combined_df = pd.concat([
        original_df[['ktas01', 'emtrt_rust', 'original_er_stay', 'original_admit_time']],
        synthetic_gaps[['synthetic_er_stay', 'synthetic_admit_time']]
    ], axis=1)
    
    # Compare distributions
    comparison_results = compare_distributions(original_df, synthetic_gaps)
    
    # Save results
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Save comparison results as JSON
    import json
    with open(output_dir / 'time_gap_test_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    logger.info(f"Saved test results to {output_dir / 'time_gap_test_results.json'}")
    
    # Create visualizations
    visualize_comparisons(original_df, synthetic_gaps)
    
    # Print summary
    print_summary(comparison_results)
    
    logger.info("Test completed successfully!")

if __name__ == "__main__":
    main()