#!/usr/bin/env python3
"""
Test Enhanced Synthetic Data Generation with Privacy Protection

This script demonstrates the enhanced synthetic data generation with
integrated privacy protection mechanisms.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import argparse
from pathlib import Path

from src.generation.enhanced_synthetic_generator import EnhancedSyntheticGenerator, PrivacyConfig
from src.privacy.privacy_validator import PrivacyValidator


def test_generation_with_different_configs(db_path: str, output_dir: str):
    """Test generation with different privacy configurations"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Test configurations
    configs = [
        {
            'name': 'baseline',
            'config': PrivacyConfig(
                k_threshold=1,
                epsilon=10.0,
                enable_k_anonymity=False,
                enable_differential_privacy=False,
                enable_generalization=False
            ),
            'n_patients': 500
        },
        {
            'name': 'moderate_privacy',
            'config': PrivacyConfig(
                k_threshold=3,
                epsilon=2.0,
                age_group_size=5,
                enable_k_anonymity=True,
                enable_differential_privacy=True,
                enable_generalization=True
            ),
            'n_patients': 500
        },
        {
            'name': 'high_privacy',
            'config': PrivacyConfig(
                k_threshold=5,
                epsilon=1.0,
                age_group_size=10,
                max_suppression_rate=0.1,
                enable_k_anonymity=True,
                enable_differential_privacy=True,
                enable_generalization=True
            ),
            'n_patients': 500
        },
        {
            'name': 'maximum_privacy',
            'config': PrivacyConfig(
                k_threshold=10,
                epsilon=0.5,
                age_group_size=20,
                geo_generalization_level='province',
                time_generalization_unit='shift',
                enable_k_anonymity=True,
                enable_differential_privacy=True,
                enable_generalization=True
            ),
            'n_patients': 1000
        }
    ]
    
    results = []
    
    for test_config in configs:
        print(f"\n{'='*60}")
        print(f"Testing configuration: {test_config['name']}")
        print(f"{'='*60}")
        
        # Create generator
        generator = EnhancedSyntheticGenerator(db_path, test_config['config'])
        
        # Generate synthetic data
        synthetic_df, validation = generator.generate(
            test_config['n_patients'],
            start_date=datetime(2017, 1, 1),
            end_date=datetime(2017, 12, 31),
            validate_privacy=True
        )
        
        # Save results
        config_output_dir = os.path.join(output_dir, test_config['name'])
        generator.save_results(synthetic_df, validation, config_output_dir)
        
        # Collect results
        if validation:
            result = {
                'config': test_config['name'],
                'n_records': len(synthetic_df),
                'k_anonymity': validation.overall_metrics.k_anonymity,
                'l_diversity': validation.overall_metrics.l_diversity,
                'risk_score': validation.overall_metrics.risk_score,
                'risk_level': validation.overall_metrics.risk_level,
                'validation_passed': validation.validation_passed
            }
            results.append(result)
            
            print(f"  Records generated: {len(synthetic_df)}")
            print(f"  K-anonymity: {validation.overall_metrics.k_anonymity}")
            print(f"  L-diversity: {validation.overall_metrics.l_diversity:.2f}")
            print(f"  Risk score: {validation.overall_metrics.risk_score:.2%}")
            print(f"  Risk level: {validation.overall_metrics.risk_level}")
            print(f"  Validation passed: {validation.validation_passed}")
    
    # Create comparison report
    comparison_df = pd.DataFrame(results)
    comparison_path = os.path.join(output_dir, 'configuration_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    
    print(f"\n{'='*60}")
    print("Configuration Comparison")
    print(f"{'='*60}")
    print(comparison_df.to_string(index=False))
    
    return comparison_df


def test_constraint_satisfaction(db_path: str, output_dir: str):
    """Test generation with specific privacy constraints"""
    
    print(f"\n{'='*60}")
    print("Testing Constraint-Based Generation")
    print(f"{'='*60}")
    
    # Define constraints
    constraints_tests = [
        {
            'name': 'min_k_5',
            'constraints': {'min_k': 5},
            'n_patients': 500
        },
        {
            'name': 'low_risk',
            'constraints': {'max_risk': 0.15},
            'n_patients': 500
        },
        {
            'name': 'strict_privacy',
            'constraints': {'min_k': 10, 'max_risk': 0.1, 'epsilon': 0.5},
            'n_patients': 1000
        }
    ]
    
    for test in constraints_tests:
        print(f"\nTesting: {test['name']}")
        print(f"Constraints: {test['constraints']}")
        
        # Create generator
        generator = EnhancedSyntheticGenerator(db_path)
        
        # Generate with constraints
        synthetic_df = generator.generate_with_constraints(
            test['n_patients'],
            test['constraints']
        )
        
        # Validate results
        validator = PrivacyValidator()
        validation = validator.validate(synthetic_df)
        
        print(f"  Records generated: {len(synthetic_df)}")
        print(f"  K-anonymity achieved: {validation.overall_metrics.k_anonymity}")
        print(f"  Risk score achieved: {validation.overall_metrics.risk_score:.2%}")
        
        # Check constraint satisfaction
        satisfied = True
        if 'min_k' in test['constraints']:
            if validation.overall_metrics.k_anonymity < test['constraints']['min_k']:
                satisfied = False
                print(f"  ❌ K-anonymity constraint not met")
        
        if 'max_risk' in test['constraints']:
            if validation.overall_metrics.risk_score > test['constraints']['max_risk']:
                satisfied = False
                print(f"  ❌ Risk constraint not met")
        
        if satisfied:
            print(f"  ✅ All constraints satisfied")


def compare_with_original(db_path: str, synthetic_df: pd.DataFrame):
    """Compare synthetic data with original data statistics"""
    
    print(f"\n{'='*60}")
    print("Statistical Comparison with Original Data")
    print(f"{'='*60}")
    
    # Load sample of original data
    import duckdb
    conn = duckdb.connect(db_path)
    
    original_df = conn.execute("""
        SELECT pat_age, pat_sex, ktas_lv, sbp, dbp, pr
        FROM nedis_data
        LIMIT 1000
    """).fetchdf()
    
    conn.close()
    
    # Compare distributions
    comparisons = []
    
    # Age distribution
    if 'pat_age' in synthetic_df.columns and 'pat_age' in original_df.columns:
        orig_mean = original_df['pat_age'].mean()
        synth_mean = synthetic_df['pat_age'].mean()
        comparisons.append({
            'metric': 'Age Mean',
            'original': f"{orig_mean:.1f}",
            'synthetic': f"{synth_mean:.1f}",
            'difference': f"{abs(orig_mean - synth_mean):.1f}"
        })
    
    # Sex distribution
    if 'pat_sex' in synthetic_df.columns and 'pat_sex' in original_df.columns:
        orig_male_pct = (original_df['pat_sex'] == 'M').mean() * 100
        synth_male_pct = (synthetic_df['pat_sex'] == 'M').mean() * 100
        comparisons.append({
            'metric': 'Male %',
            'original': f"{orig_male_pct:.1f}%",
            'synthetic': f"{synth_male_pct:.1f}%",
            'difference': f"{abs(orig_male_pct - synth_male_pct):.1f}%"
        })
    
    # KTAS distribution
    if 'ktas_lv' in synthetic_df.columns and 'ktas_lv' in original_df.columns:
        for level in [1, 2, 3, 4, 5]:
            orig_pct = (original_df['ktas_lv'] == level).mean() * 100
            synth_pct = (synthetic_df['ktas_lv'] == level).mean() * 100
            comparisons.append({
                'metric': f'KTAS {level} %',
                'original': f"{orig_pct:.1f}%",
                'synthetic': f"{synth_pct:.1f}%",
                'difference': f"{abs(orig_pct - synth_pct):.1f}%"
            })
    
    # Vital signs
    for col in ['sbp', 'dbp', 'pr']:
        if col in synthetic_df.columns and col in original_df.columns:
            orig_mean = pd.to_numeric(original_df[col], errors='coerce').mean()
            synth_mean = pd.to_numeric(synthetic_df[col], errors='coerce').mean()
            if not pd.isna(orig_mean) and not pd.isna(synth_mean):
                comparisons.append({
                    'metric': f'{col.upper()} Mean',
                    'original': f"{orig_mean:.1f}",
                    'synthetic': f"{synth_mean:.1f}",
                    'difference': f"{abs(orig_mean - synth_mean):.1f}"
                })
    
    comparison_df = pd.DataFrame(comparisons)
    print(comparison_df.to_string(index=False))
    
    return comparison_df


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='Test enhanced synthetic data generation')
    parser.add_argument('--db', type=str, default='data/nedis_data.duckdb',
                       help='Path to database')
    parser.add_argument('--output-dir', type=str, default='outputs/privacy_test',
                       help='Output directory')
    parser.add_argument('--test-configs', action='store_true',
                       help='Test different privacy configurations')
    parser.add_argument('--test-constraints', action='store_true',
                       help='Test constraint satisfaction')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run a quick test with minimal data')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("="*60)
    print("Enhanced Synthetic Data Generation Test")
    print("Privacy-Preserving NEDIS Data Synthesis")
    print("="*60)
    
    if args.quick_test:
        # Quick test with small sample
        print("\nRunning quick test...")
        
        config = PrivacyConfig(
            k_threshold=5,
            epsilon=1.0,
            enable_k_anonymity=True,
            enable_differential_privacy=True,
            enable_generalization=True
        )
        
        generator = EnhancedSyntheticGenerator(args.db, config)
        
        synthetic_df, validation = generator.generate(
            100,  # Small number for quick test
            start_date=datetime(2017, 1, 1),
            end_date=datetime(2017, 1, 31),
            validate_privacy=True
        )
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        generator.save_results(synthetic_df, validation, args.output_dir)
        
        print(f"\nQuick test complete:")
        print(f"  Generated {len(synthetic_df)} records")
        if validation:
            print(f"  Risk level: {validation.overall_metrics.risk_level}")
            print(f"  K-anonymity: {validation.overall_metrics.k_anonymity}")
            print(f"  Validation passed: {validation.validation_passed}")
        
        # Compare with original
        compare_with_original(args.db, synthetic_df)
        
    else:
        # Full tests
        if args.test_configs:
            test_generation_with_different_configs(args.db, args.output_dir)
        
        if args.test_constraints:
            test_constraint_satisfaction(args.db, args.output_dir)
        
        if not args.test_configs and not args.test_constraints:
            # Default test
            print("\nRunning default privacy-enhanced generation...")
            
            config = PrivacyConfig(
                k_threshold=5,
                epsilon=1.0,
                age_group_size=5,
                enable_k_anonymity=True,
                enable_differential_privacy=True,
                enable_generalization=True
            )
            
            generator = EnhancedSyntheticGenerator(args.db, config)
            
            synthetic_df, validation = generator.generate(
                500,
                start_date=datetime(2017, 1, 1),
                end_date=datetime(2017, 12, 31),
                validate_privacy=True
            )
            
            # Save results
            os.makedirs(args.output_dir, exist_ok=True)
            generator.save_results(synthetic_df, validation, args.output_dir)
            
            print(f"\nGeneration complete:")
            print(f"  Generated {len(synthetic_df)} records")
            if validation:
                print(f"  Risk level: {validation.overall_metrics.risk_level}")
                print(f"  K-anonymity: {validation.overall_metrics.k_anonymity}")
                print(f"  L-diversity: {validation.overall_metrics.l_diversity:.2f}")
                print(f"  Risk score: {validation.overall_metrics.risk_score:.2%}")
                print(f"  Validation passed: {validation.validation_passed}")
            
            # Compare with original
            compare_with_original(args.db, synthetic_df)
    
    print(f"\nResults saved to: {args.output_dir}")
    print("\n✅ Test completed successfully!")


if __name__ == '__main__':
    main()