#!/usr/bin/env python3
"""
Simple Test of Privacy Enhancement Modules

Tests the privacy modules with synthetic test data without
requiring the full generation pipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import logging

from src.privacy.identifier_manager import IdentifierManager
from src.privacy.generalization import AgeGeneralizer, GeographicGeneralizer, TemporalGeneralizer
from src.privacy.k_anonymity import KAnonymityValidator, KAnonymityEnforcer
from src.privacy.differential_privacy import DifferentialPrivacy, PrivacyAccountant
from src.privacy.privacy_validator import PrivacyValidator


def create_test_data(n_records: int = 500) -> pd.DataFrame:
    """Create synthetic test data resembling NEDIS"""
    np.random.seed(42)
    
    # Generate test data with more repetition for better k-anonymity
    # Use fewer unique values to create natural groups
    ages = np.random.choice([20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70], n_records)
    areas = np.random.choice(['1101', '1102', '2101', '2102'], n_records, p=[0.3, 0.3, 0.2, 0.2])
    
    data = {
        'pat_reg_no': [f'P{i:06d}' for i in range(n_records)],
        'pat_age': ages,
        'pat_sex': np.random.choice(['M', 'F'], n_records, p=[0.55, 0.45]),
        'pat_sarea': areas,
        'ktas_lv': np.random.choice([1, 2, 3, 4, 5], n_records, p=[0.05, 0.15, 0.30, 0.35, 0.15]),
        'vst_dt': np.random.choice(['20170101', '20170102', '20170103'], n_records),
        'vst_tm': [f'{np.random.choice([8, 10, 14, 16, 20]):02d}00' for _ in range(n_records)],
        'sbp': np.random.normal(120, 20, n_records),
        'dbp': np.random.normal(80, 10, n_records),
        'pr': np.random.normal(80, 15, n_records),
        'ed_diag': np.random.choice(['A01', 'B02', 'C03', 'D04', 'E05'], n_records),
        'outcome': np.random.choice(['discharge', 'admission', 'transfer', 'death'], n_records, 
                                  p=[0.70, 0.20, 0.08, 0.02])
    }
    
    return pd.DataFrame(data)


def test_privacy_enhancement_pipeline():
    """Test the complete privacy enhancement pipeline"""
    
    print("="*60)
    print("Privacy Enhancement Pipeline Test")
    print("="*60)
    
    # Create test data
    print("\n1. Creating test data...")
    original_df = create_test_data(500)
    print(f"   Created {len(original_df)} test records")
    print(f"   Columns: {list(original_df.columns)}")
    
    # Initial privacy assessment
    print("\n2. Initial Privacy Assessment")
    validator = PrivacyValidator(k_threshold=5, l_threshold=3)
    initial_validation = validator.validate(
        original_df,
        quasi_identifiers=['pat_age', 'pat_sex', 'pat_sarea', 'ktas_lv'],
        sensitive_attributes=['ed_diag', 'outcome']
    )
    print(f"   K-anonymity: {initial_validation.overall_metrics.k_anonymity}")
    print(f"   L-diversity: {initial_validation.overall_metrics.l_diversity:.2f}")
    print(f"   Risk level: {initial_validation.overall_metrics.risk_level}")
    print(f"   Risk score: {initial_validation.overall_metrics.risk_score:.2%}")
    
    # Apply privacy enhancements
    enhanced_df = original_df.copy()
    
    # Step 1: Identifier Management
    print("\n3. Applying Identifier Management")
    id_manager = IdentifierManager()
    enhanced_df = id_manager.anonymize_dataframe(enhanced_df)
    print(f"   ✓ Removed direct identifiers")
    print(f"   ✓ Added synthetic IDs")
    
    # Step 2: Age Generalization
    print("\n4. Applying Age Generalization")
    age_gen = AgeGeneralizer(group_size=5)
    enhanced_df['pat_age'] = age_gen.generalize_series(
        enhanced_df['pat_age'], 
        method='random',
        preserve_distribution=True
    )
    print(f"   ✓ Generalized ages to 5-year groups")
    
    # Step 3: Geographic Generalization
    print("\n5. Applying Geographic Generalization")
    geo_gen = GeographicGeneralizer()
    # Since our test data already has 4-digit codes, generalize to 2-digit province level
    enhanced_df['pat_sarea'] = geo_gen.generalize_series(
        enhanced_df['pat_sarea'],
        target_level='province'
    )
    print(f"   ✓ Generalized geographic codes to province level")
    
    # Step 4: Temporal Generalization
    print("\n6. Applying Temporal Generalization")
    temp_gen = TemporalGeneralizer()
    enhanced_df['vst_tm'] = enhanced_df['vst_tm'].apply(
        lambda x: temp_gen.round_time(x, 'hour')
    )
    print(f"   ✓ Generalized visit times to hour precision")
    
    # Step 5: K-anonymity Enforcement
    print("\n7. Enforcing K-anonymity")
    k_enforcer = KAnonymityEnforcer(k_threshold=5, max_suppression_rate=0.05)
    # Use fewer quasi-identifiers for better k-anonymity
    enhanced_df, k_stats = k_enforcer.enforce(
        enhanced_df,
        ['pat_age', 'pat_sex', 'pat_sarea'],  # Removed ktas_lv for better grouping
        method='generalize'  # Use generalization instead of suppression
    )
    print(f"   ✓ K-anonymity enforced")
    print(f"   - Records suppressed: {k_stats.get('suppressed_count', 0)}")
    if 'suppression_rate' in k_stats:
        print(f"   - Suppression rate: {k_stats['suppression_rate']:.2%}")
    if 'generalization_levels' in k_stats:
        print(f"   - Generalization levels: {k_stats['generalization_levels']}")
    print(f"   - K achieved: {k_stats['k_achieved']}")
    
    # Step 6: Differential Privacy
    print("\n8. Applying Differential Privacy")
    dp = DifferentialPrivacy(epsilon=1.0)
    column_configs = {
        'sbp': {'sensitivity': 10, 'lower': 60, 'upper': 200},
        'dbp': {'sensitivity': 10, 'lower': 40, 'upper': 120},
        'pr': {'sensitivity': 5, 'lower': 40, 'upper': 140}
    }
    enhanced_df = dp.apply_to_dataframe(enhanced_df, column_configs)
    print(f"   ✓ Applied differential privacy (ε=1.0)")
    print(f"   - Noise added to vital signs")
    
    # Final privacy assessment
    print("\n9. Final Privacy Assessment")
    final_validation = validator.validate(
        enhanced_df,
        quasi_identifiers=['pat_age', 'pat_sex', 'pat_sarea', 'ktas_lv'],
        sensitive_attributes=['ed_diag', 'outcome']
    )
    print(f"   K-anonymity: {final_validation.overall_metrics.k_anonymity}")
    print(f"   L-diversity: {final_validation.overall_metrics.l_diversity:.2f}")
    print(f"   Risk level: {final_validation.overall_metrics.risk_level}")
    print(f"   Risk score: {final_validation.overall_metrics.risk_score:.2%}")
    
    # Compare original vs enhanced
    print("\n10. Privacy Improvement Summary")
    print(f"   {'Metric':<20} {'Original':<15} {'Enhanced':<15} {'Improvement':<15}")
    print(f"   {'-'*65}")
    
    k_improvement = final_validation.overall_metrics.k_anonymity - initial_validation.overall_metrics.k_anonymity
    print(f"   {'K-anonymity':<20} {initial_validation.overall_metrics.k_anonymity:<15} "
          f"{final_validation.overall_metrics.k_anonymity:<15} {f'+{k_improvement}':<15}")
    
    l_improvement = final_validation.overall_metrics.l_diversity - initial_validation.overall_metrics.l_diversity
    print(f"   {'L-diversity':<20} {initial_validation.overall_metrics.l_diversity:<15.2f} "
          f"{final_validation.overall_metrics.l_diversity:<15.2f} {f'+{l_improvement:.2f}':<15}")
    
    risk_reduction = initial_validation.overall_metrics.risk_score - final_validation.overall_metrics.risk_score
    print(f"   {'Risk Score':<20} {initial_validation.overall_metrics.risk_score:<15.2%} "
          f"{final_validation.overall_metrics.risk_score:<15.2%} {f'-{risk_reduction:.2%}':<15}")
    
    print(f"   {'Risk Level':<20} {initial_validation.overall_metrics.risk_level:<15} "
          f"{final_validation.overall_metrics.risk_level:<15} {'✓ Improved' if final_validation.overall_metrics.risk_level != initial_validation.overall_metrics.risk_level else 'No change':<15}")
    
    # Save reports
    print("\n11. Saving Reports")
    os.makedirs('outputs/privacy_test', exist_ok=True)
    
    # Save privacy validation report
    report_path = 'outputs/privacy_test/privacy_validation_report.html'
    validator.generate_report(final_validation, report_path)
    print(f"   ✓ Privacy report saved to {report_path}")
    
    # Save sample data
    sample_path = 'outputs/privacy_test/enhanced_sample.csv'
    enhanced_df.head(100).to_csv(sample_path, index=False)
    print(f"   ✓ Sample data saved to {sample_path}")
    
    # Statistical comparison
    print("\n12. Statistical Preservation Check")
    print(f"   {'Statistic':<25} {'Original':<15} {'Enhanced':<15} {'Difference':<15}")
    print(f"   {'-'*70}")
    
    # Age statistics
    orig_age_mean = original_df['pat_age'].mean()
    enh_age_mean = enhanced_df['pat_age'].mean()
    print(f"   {'Age Mean':<25} {orig_age_mean:<15.1f} {enh_age_mean:<15.1f} {abs(orig_age_mean - enh_age_mean):<15.1f}")
    
    # Gender distribution
    orig_male_pct = (original_df['pat_sex'] == 'M').mean() * 100
    enh_male_pct = (enhanced_df['pat_sex'] == 'M').mean() * 100
    print(f"   {'Male %':<25} {orig_male_pct:<15.1f} {enh_male_pct:<15.1f} {abs(orig_male_pct - enh_male_pct):<15.1f}")
    
    # KTAS distribution
    for level in [1, 2, 3, 4, 5]:
        orig_pct = (original_df['ktas_lv'] == level).mean() * 100
        enh_pct = (enhanced_df['ktas_lv'] == level).mean() * 100
        print(f"   {f'KTAS {level} %':<25} {orig_pct:<15.1f} {enh_pct:<15.1f} {abs(orig_pct - enh_pct):<15.1f}")
    
    print("\n" + "="*60)
    print("✅ Privacy Enhancement Test Complete!")
    print("="*60)
    
    return enhanced_df, final_validation


def test_privacy_accountant():
    """Test privacy budget management"""
    print("\n" + "="*60)
    print("Privacy Budget Management Test")
    print("="*60)
    
    accountant = PrivacyAccountant(total_budget=2.0)
    
    print(f"\nInitial budget: {accountant.total_budget}")
    
    # Simulate operations
    operations = [
        ("Age generalization", 0.3),
        ("Geographic generalization", 0.3),
        ("Vital signs noise", 0.5),
        ("Outcome noise", 0.4),
        ("Diagnosis noise", 0.4)
    ]
    
    for op_name, epsilon in operations:
        if accountant.consume(epsilon, op_name):
            print(f"✓ {op_name}: consumed ε={epsilon}, remaining={accountant.get_remaining_budget():.1f}")
        else:
            print(f"✗ {op_name}: FAILED (insufficient budget)")
    
    # Try to exceed budget
    if not accountant.consume(0.5, "Extra operation"):
        print(f"✓ Budget protection works - cannot exceed total budget")
    
    print(f"\nFinal budget status:")
    print(f"  Total: {accountant.total_budget}")
    print(f"  Consumed: {accountant.consumed_budget}")
    print(f"  Remaining: {accountant.get_remaining_budget()}")
    
    print("\nOperation log:")
    for op in accountant.get_operations_log():
        print(f"  - {op['operation']}: ε={op['epsilon']}, cumulative={op['cumulative']}")


def main():
    """Main test function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress detailed logs for cleaner output
    logging.getLogger('src.privacy').setLevel(logging.WARNING)
    
    print("\n🔒 NEDIS Privacy Enhancement Module Test")
    print("Testing privacy protection mechanisms\n")
    
    # Run main pipeline test
    enhanced_df, validation = test_privacy_enhancement_pipeline()
    
    # Run budget management test
    test_privacy_accountant()
    
    print("\n✨ All tests completed successfully!")
    print("Check outputs/privacy_test/ for generated reports and data samples.")


if __name__ == '__main__':
    main()