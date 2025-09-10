"""
Test Suite for Privacy Enhancement Modules

Tests all privacy components including k-anonymity, differential privacy,
generalization, and identifier management.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.privacy.identifier_manager import IdentifierManager
from src.privacy.generalization import AgeGeneralizer, GeographicGeneralizer, TemporalGeneralizer
from src.privacy.k_anonymity import KAnonymityValidator, KAnonymityEnforcer
from src.privacy.differential_privacy import DifferentialPrivacy, PrivacyAccountant
from src.privacy.privacy_validator import PrivacyValidator


class TestIdentifierManager:
    """Test identifier management functionality"""
    
    def test_unique_id_generation(self):
        """Test that generated IDs are unique"""
        manager = IdentifierManager()
        ids = [manager.generate_unique_id() for _ in range(100)]
        
        # Check uniqueness
        assert len(ids) == len(set(ids))
        
        # Check format
        for id in ids:
            assert id.startswith("SYN_")
            assert len(id.split("_")) == 3
    
    def test_anonymize_dataframe(self):
        """Test dataframe anonymization"""
        manager = IdentifierManager()
        
        # Create test dataframe with direct identifiers
        df = pd.DataFrame({
            'pat_reg_no': ['P001', 'P002', 'P003'],
            'pat_brdt': ['19800101', '19900202', '19700303'],
            'pat_age': [37, 27, 47],
            'pat_sex': ['M', 'F', 'M'],
            'index_key': [1, 2, 3]
        })
        
        # Anonymize
        anon_df = manager.anonymize_dataframe(df)
        
        # Check direct identifiers removed
        assert 'pat_reg_no' not in anon_df.columns
        assert 'index_key' not in anon_df.columns
        assert 'pat_brdt' not in anon_df.columns
        
        # Check synthetic ID added
        assert 'synthetic_id' in anon_df.columns
        assert len(anon_df['synthetic_id'].unique()) == len(anon_df)
    
    def test_validation(self):
        """Test anonymization validation"""
        manager = IdentifierManager()
        
        # Create properly anonymized dataframe
        df = pd.DataFrame({
            'synthetic_id': ['SYN_001', 'SYN_002'],
            'pat_age': [30, 40],
            'pat_sex': ['M', 'F']
        })
        
        validation = manager.validate_anonymization(df)
        assert validation['valid']
        assert validation['has_synthetic_id']


class TestAgeGeneralizer:
    """Test age generalization"""
    
    def test_age_generalization(self):
        """Test basic age generalization"""
        generalizer = AgeGeneralizer(group_size=10)
        
        # Test single age
        age = 25
        generalized = generalizer.generalize(age, method='lower')
        assert generalized == 20
        
        generalized = generalizer.generalize(age, method='upper')
        assert generalized == 29
        
        generalized = generalizer.generalize(age, method='center')
        assert generalized == 24
    
    def test_age_series_generalization(self):
        """Test generalization of age series"""
        generalizer = AgeGeneralizer(group_size=5)
        
        ages = pd.Series([22, 23, 24, 26, 27, 31, 32, 45, 67, 89])
        generalized = generalizer.generalize_series(ages, method='lower')
        
        # Check all ages are generalized to group starts
        assert all(generalized % 5 == 0)
    
    def test_special_age_groups(self):
        """Test special age group handling"""
        generalizer = AgeGeneralizer(
            group_size=10,
            special_groups={(0, 2): 1, (90, 120): 30}
        )
        
        # Infant should remain unchanged
        assert generalizer.generalize(1, method='center') == 1
        
        # Elderly should be grouped together
        generalized_95 = generalizer.generalize(95, method='center')
        generalized_100 = generalizer.generalize(100, method='center')
        assert abs(generalized_95 - generalized_100) <= 30


class TestGeographicGeneralizer:
    """Test geographic generalization"""
    
    def test_region_code_generalization(self):
        """Test region code generalization"""
        generalizer = GeographicGeneralizer()
        
        # Test different levels
        region = "110101"  # 6-digit region code
        
        assert generalizer.generalize(region, 'province') == "11"
        assert generalizer.generalize(region, 'district') == "1101"
        assert generalizer.generalize(region, 'detail') == "110101"
    
    def test_rare_region_suppression(self):
        """Test suppression of rare regions"""
        generalizer = GeographicGeneralizer()
        
        regions = pd.Series(['11', '11', '11', '21', '21', '31', '41'])
        suppressed = generalizer.suppress_rare_regions(regions, min_count=2)
        
        # Regions 31 and 41 should be suppressed
        assert (suppressed == 'OTHER').sum() == 2


class TestKAnonymity:
    """Test k-anonymity validation and enforcement"""
    
    def test_k_anonymity_validation(self):
        """Test k-anonymity validation"""
        validator = KAnonymityValidator(k_threshold=3)
        
        # Create test data with varying group sizes
        df = pd.DataFrame({
            'age': [20, 20, 20, 30, 30, 40],
            'sex': ['M', 'M', 'M', 'F', 'F', 'M'],
            'region': ['A', 'A', 'A', 'B', 'B', 'C']
        })
        
        result = validator.validate(df, ['age', 'sex', 'region'])
        
        # Minimum group size should be 1 (40, M, C)
        assert result.k_value == 1
        assert not result.satisfied
        assert result.num_violations == 1
    
    def test_k_anonymity_enforcement(self):
        """Test k-anonymity enforcement through suppression"""
        enforcer = KAnonymityEnforcer(k_threshold=2)
        
        # Create test data
        df = pd.DataFrame({
            'age': [20, 20, 30, 40, 50],
            'sex': ['M', 'M', 'F', 'M', 'F'],
            'value': [1, 2, 3, 4, 5]
        })
        
        enforced_df, stats = enforcer.enforce(df, ['age', 'sex'], method='suppress')
        
        # Check that k-anonymity is satisfied
        validator = KAnonymityValidator(k_threshold=2)
        result = validator.validate(enforced_df, ['age', 'sex'])
        assert result.satisfied or len(enforced_df) < len(df)


class TestDifferentialPrivacy:
    """Test differential privacy mechanisms"""
    
    def test_laplace_noise(self):
        """Test Laplace noise addition"""
        dp = DifferentialPrivacy(epsilon=1.0)
        
        # Test single value
        value = 100
        noisy = dp.add_laplace_noise(value, sensitivity=1)
        
        # Value should be changed but reasonable
        assert noisy != value
        assert abs(noisy - value) < 20  # With high probability
    
    def test_gaussian_noise(self):
        """Test Gaussian noise addition"""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        
        # Test array
        values = np.array([100, 200, 300])
        noisy = dp.add_gaussian_noise(values, sensitivity=10)
        
        # Check shape preserved
        assert noisy.shape == values.shape
        
        # Values should be changed
        assert not np.array_equal(noisy, values)
    
    def test_exponential_mechanism(self):
        """Test exponential mechanism for discrete selection"""
        dp = DifferentialPrivacy(epsilon=2.0)
        
        candidates = ['A', 'B', 'C', 'D']
        scores = [10, 9, 8, 7]  # A has highest utility
        
        # Run multiple times to check probabilistic selection
        selections = []
        for _ in range(100):
            selected = dp.exponential_mechanism(candidates, scores, sensitivity=1)
            selections.append(selected)
        
        # A should be selected most often
        from collections import Counter
        counts = Counter(selections)
        assert counts.most_common(1)[0][0] == 'A'
    
    def test_privacy_budget_tracking(self):
        """Test privacy budget consumption tracking"""
        dp = DifferentialPrivacy(epsilon=1.0)
        
        initial_budget = dp.consumed_budget
        
        # Consume budget
        dp.add_laplace_noise(100, sensitivity=1)
        
        # Check budget increased
        assert dp.consumed_budget > initial_budget
        
        # Get status
        status = dp.get_privacy_budget_status()
        assert status['consumed_budget'] > 0
        assert status['remaining_budget'] < status['total_budget']


class TestPrivacyAccountant:
    """Test privacy budget accountant"""
    
    def test_budget_consumption(self):
        """Test budget consumption tracking"""
        accountant = PrivacyAccountant(total_budget=2.0)
        
        # Consume some budget
        assert accountant.consume(0.5, "operation1")
        assert accountant.consume(1.0, "operation2")
        
        # Should have 0.5 remaining
        assert accountant.get_remaining_budget() == 0.5
        
        # Should fail to consume more than available
        assert not accountant.consume(1.0, "operation3")
    
    def test_operations_log(self):
        """Test operations logging"""
        accountant = PrivacyAccountant(total_budget=2.0)
        
        accountant.consume(0.5, "test_op")
        
        log = accountant.get_operations_log()
        assert len(log) == 1
        assert log[0]['operation'] == "test_op"
        assert log[0]['epsilon'] == 0.5


class TestPrivacyValidator:
    """Test comprehensive privacy validation"""
    
    def test_comprehensive_validation(self):
        """Test full privacy validation"""
        validator = PrivacyValidator(k_threshold=2, l_threshold=2, epsilon=1.0)
        
        # Create test data
        df = pd.DataFrame({
            'age': [20, 20, 30, 30, 40, 40],
            'sex': ['M', 'M', 'F', 'F', 'M', 'M'],
            'region': ['A', 'A', 'B', 'B', 'C', 'C'],
            'diagnosis': ['D1', 'D2', 'D1', 'D2', 'D1', 'D2']
        })
        
        result = validator.validate(
            df,
            quasi_identifiers=['age', 'sex', 'region'],
            sensitive_attributes=['diagnosis']
        )
        
        # Check result structure
        assert result.overall_metrics is not None
        assert result.k_anonymity_result is not None
        assert result.l_diversity_scores is not None
        assert result.attribute_risks is not None
        assert result.privacy_guarantees is not None
        
        # Check metrics
        assert result.overall_metrics.k_anonymity == 2
        assert result.overall_metrics.l_diversity >= 2
        assert result.overall_metrics.risk_level in ['LOW', 'MEDIUM', 'HIGH']
    
    def test_report_generation(self):
        """Test HTML report generation"""
        validator = PrivacyValidator()
        
        # Create simple test data
        df = pd.DataFrame({
            'age': [20, 30, 40],
            'sex': ['M', 'F', 'M']
        })
        
        result = validator.validate(df, quasi_identifiers=['age', 'sex'])
        
        # Generate report
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            report_html = validator.generate_report(result, f.name)
            
            # Check report contains key elements
            assert 'Privacy Validation Report' in report_html
            assert 'K-Anonymity' in report_html
            assert 'Risk Assessment' in report_html
            
            # Check file was created
            assert os.path.exists(f.name)
            os.unlink(f.name)


class TestIntegration:
    """Integration tests for privacy modules working together"""
    
    def test_full_privacy_pipeline(self):
        """Test complete privacy enhancement pipeline"""
        # Create realistic test data
        np.random.seed(42)
        n_records = 100
        
        df = pd.DataFrame({
            'pat_reg_no': [f'P{i:04d}' for i in range(n_records)],
            'pat_age': np.random.randint(18, 80, n_records),
            'pat_sex': np.random.choice(['M', 'F'], n_records),
            'pat_sarea': np.random.choice(['110101', '110102', '210101', '210102'], n_records),
            'ktas_lv': np.random.choice([1, 2, 3, 4, 5], n_records),
            'sbp': np.random.normal(120, 20, n_records),
            'dbp': np.random.normal(80, 10, n_records)
        })
        
        # Step 1: Identifier management
        id_manager = IdentifierManager()
        df = id_manager.anonymize_dataframe(df)
        assert 'pat_reg_no' not in df.columns
        assert 'synthetic_id' in df.columns
        
        # Step 2: Generalization
        age_gen = AgeGeneralizer(group_size=5)
        df['pat_age'] = age_gen.generalize_series(df['pat_age'], method='random')
        
        geo_gen = GeographicGeneralizer()
        df['pat_sarea'] = geo_gen.generalize_series(df['pat_sarea'], 'district')
        
        # Step 3: K-anonymity enforcement
        k_enforcer = KAnonymityEnforcer(k_threshold=3)
        df, _ = k_enforcer.enforce(df, ['pat_age', 'pat_sex', 'pat_sarea'])
        
        # Step 4: Differential privacy
        dp = DifferentialPrivacy(epsilon=1.0)
        column_configs = {
            'sbp': {'sensitivity': 10, 'lower': 60, 'upper': 200},
            'dbp': {'sensitivity': 10, 'lower': 40, 'upper': 120}
        }
        df = dp.apply_to_dataframe(df, column_configs)
        
        # Step 5: Validate privacy
        validator = PrivacyValidator(k_threshold=3)
        result = validator.validate(
            df,
            quasi_identifiers=['pat_age', 'pat_sex', 'pat_sarea']
        )
        
        # Check privacy guarantees
        assert result.overall_metrics.k_anonymity >= 3
        assert result.overall_metrics.risk_level in ['LOW', 'MEDIUM', 'HIGH']


def test_no_hardcoded_distributions():
    """Verify no hardcoded distributions in privacy modules"""
    # This test ensures we're not hardcoding probability distributions
    # All distributions should be learned from data or configurable
    
    import inspect
    import re
    
    # Modules to check
    modules = [
        'src/privacy/differential_privacy.py',
        'src/privacy/generalization.py',
        'src/privacy/k_anonymity.py'
    ]
    
    hardcoded_patterns = [
        r'distribution\s*=\s*\{.*\}',  # Hardcoded dict distributions
        r'weights\s*=\s*\[.*\]',        # Hardcoded weight arrays
        r'probabilities\s*=\s*\[.*\]'   # Hardcoded probability arrays
    ]
    
    for module_path in modules:
        if os.path.exists(module_path):
            with open(module_path, 'r') as f:
                content = f.read()
                
            for pattern in hardcoded_patterns:
                matches = re.findall(pattern, content)
                # Filter out legitimate uses (like empty initializations)
                matches = [m for m in matches if not re.match(r'.*\[\s*\]', m)]
                assert len(matches) == 0, f"Found hardcoded pattern in {module_path}: {matches}"


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])