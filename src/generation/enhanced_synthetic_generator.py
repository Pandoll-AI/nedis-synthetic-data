"""
Enhanced Synthetic Data Generator with Privacy Protection

Integrates all privacy mechanisms into the synthetic data generation pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import json
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..privacy.identifier_manager import IdentifierManager
from ..privacy.generalization import AgeGeneralizer, GeographicGeneralizer, TemporalGeneralizer
from ..privacy.k_anonymity import KAnonymityValidator, KAnonymityEnforcer
from ..privacy.differential_privacy import DifferentialPrivacy, PrivacyAccountant
from ..privacy.privacy_validator import PrivacyValidator, PrivacyValidationResult
from ..vectorized.patient_generator import VectorizedPatientGenerator
from ..vectorized.temporal_assigner import TemporalPatternAssigner
from ..temporal.comprehensive_time_gap_synthesizer import ComprehensiveTimeGapSynthesizer
from ..core.database import DatabaseManager
from ..core.config import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class PrivacyConfig:
    """Configuration for privacy enhancement"""
    # K-anonymity settings
    k_threshold: int = 5
    max_suppression_rate: float = 0.05
    
    # L-diversity settings
    l_threshold: int = 3
    
    # Differential privacy settings
    epsilon: float = 1.0
    delta: float = 1e-5
    
    # Generalization settings
    age_group_size: int = 5
    geo_generalization_level: str = 'district'
    time_generalization_unit: str = 'hour'
    
    # Risk thresholds
    max_risk_score: float = 0.2
    
    # Feature flags
    enable_k_anonymity: bool = True
    enable_differential_privacy: bool = True
    enable_generalization: bool = True
    enable_identifier_management: bool = True


class EnhancedSyntheticGenerator:
    """
    Enhanced synthetic data generator with integrated privacy protection
    """
    
    def __init__(self, db_path: str, config: Optional[PrivacyConfig] = None):
        """
        Initialize enhanced generator
        
        Args:
            db_path: Path to database
            config: Privacy configuration
        """
        self.db_path = db_path
        self.config = config or PrivacyConfig()
        
        # Initialize database and config managers
        self.db_manager = DatabaseManager(db_path)
        self.config_manager = ConfigManager()
        
        # Initialize base generators
        self.base_generator = VectorizedPatientGenerator(self.db_manager, self.config_manager)
        self.temporal_assigner = TemporalPatternAssigner(self.db_manager, self.config_manager)
        # Create a simple config dict for time gap synthesizer
        time_gap_config = {
            'ktas_time_gaps': {
                1: {'mean': 30, 'std': 15},
                2: {'mean': 60, 'std': 30},
                3: {'mean': 120, 'std': 60},
                4: {'mean': 180, 'std': 90},
                5: {'mean': 240, 'std': 120}
            }
        }
        self.time_gap_synthesizer = ComprehensiveTimeGapSynthesizer(db_path, time_gap_config)
        
        # Initialize privacy components
        self.id_manager = IdentifierManager()
        self.age_generalizer = AgeGeneralizer(group_size=self.config.age_group_size)
        self.geo_generalizer = GeographicGeneralizer()
        self.temporal_generalizer = TemporalGeneralizer()
        self.k_enforcer = KAnonymityEnforcer(
            k_threshold=self.config.k_threshold,
            max_suppression_rate=self.config.max_suppression_rate
        )
        self.dp = DifferentialPrivacy(
            epsilon=self.config.epsilon,
            delta=self.config.delta
        )
        self.privacy_accountant = PrivacyAccountant(total_budget=self.config.epsilon)
        self.privacy_validator = PrivacyValidator(
            k_threshold=self.config.k_threshold,
            l_threshold=self.config.l_threshold,
            epsilon=self.config.epsilon,
            max_risk_score=self.config.max_risk_score
        )
        
        logger.info(f"Initialized enhanced generator with privacy config: "
                   f"k={self.config.k_threshold}, ε={self.config.epsilon}")
    
    def generate(self, n_patients: int,
                start_date: Optional[datetime] = None,
                end_date: Optional[datetime] = None,
                validate_privacy: bool = True) -> Tuple[pd.DataFrame, PrivacyValidationResult]:
        """
        Generate synthetic data with privacy protection
        
        Args:
            n_patients: Number of patients to generate
            start_date: Start date for temporal data
            end_date: End date for temporal data
            validate_privacy: Whether to validate privacy after generation
            
        Returns:
            Tuple of (synthetic data, privacy validation result)
        """
        logger.info(f"Starting enhanced generation for {n_patients} patients")
        
        # Phase 1: Generate base synthetic data
        logger.info("Phase 1: Generating base synthetic data")
        synthetic_df = self._generate_base_data(n_patients, start_date, end_date)
        
        # Phase 2: Apply identifier management
        if self.config.enable_identifier_management:
            logger.info("Phase 2: Managing identifiers")
            synthetic_df = self._manage_identifiers(synthetic_df)
        
        # Phase 3: Apply generalization
        if self.config.enable_generalization:
            logger.info("Phase 3: Applying generalization")
            synthetic_df = self._apply_generalization(synthetic_df)
        
        # Phase 4: Enforce k-anonymity
        if self.config.enable_k_anonymity:
            logger.info("Phase 4: Enforcing k-anonymity")
            synthetic_df = self._enforce_k_anonymity(synthetic_df)
        
        # Phase 5: Apply differential privacy
        if self.config.enable_differential_privacy:
            logger.info("Phase 5: Applying differential privacy")
            synthetic_df = self._apply_differential_privacy(synthetic_df)
        
        # Phase 6: Validate privacy
        validation_result = None
        if validate_privacy:
            logger.info("Phase 6: Validating privacy")
            validation_result = self.privacy_validator.validate(synthetic_df)
            
            # Log validation results
            logger.info(f"Privacy validation complete:")
            logger.info(f"  - Risk level: {validation_result.overall_metrics.risk_level}")
            logger.info(f"  - K-anonymity: {validation_result.overall_metrics.k_anonymity}")
            logger.info(f"  - Validation passed: {validation_result.validation_passed}")
        
        # Phase 7: Post-processing
        synthetic_df = self._post_process(synthetic_df)
        
        logger.info(f"Enhanced generation complete. Generated {len(synthetic_df)} records")
        
        return synthetic_df, validation_result
    
    def _generate_base_data(self, n_patients: int,
                          start_date: Optional[datetime],
                          end_date: Optional[datetime]) -> pd.DataFrame:
        """Generate base synthetic data"""
        from ..vectorized.patient_generator import PatientGenerationConfig
        
        # Create generation config
        gen_config = PatientGenerationConfig(
            total_records=n_patients,
            batch_size=min(n_patients, 10000),
            random_seed=42
        )
        
        # Generate patient characteristics
        patients_df = self.base_generator.generate_all_patients(gen_config)
        
        # Assign temporal patterns if generator didn't
        if 'vst_dt' not in patients_df.columns and start_date and end_date:
            patients_df = self.temporal_assigner.assign_all_timestamps(
                patients_df, gen_config
            )
        
        # Generate time gaps if not already present
        if 'ocur_dt' in patients_df.columns or 'ocur_tm' in patients_df.columns:
            patients_df = self.time_gap_synthesizer.generate_all_time_gaps(patients_df)
        
        return patients_df
    
    def _manage_identifiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply identifier management"""
        # Remove direct identifiers and add synthetic IDs
        df = self.id_manager.anonymize_dataframe(
            df,
            id_column='synthetic_id',
            preserve_columns=[]  # Don't preserve any direct identifiers
        )
        
        # Validate anonymization
        validation = self.id_manager.validate_anonymization(df)
        if not validation['valid']:
            logger.warning("Identifier anonymization validation failed")
        
        return df
    
    def _apply_generalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply generalization to quasi-identifiers"""
        # Age generalization
        if 'pat_age' in df.columns:
            logger.info(f"Generalizing age with group size {self.config.age_group_size}")
            df['pat_age'] = self.age_generalizer.generalize_series(
                df['pat_age'],
                method='random',
                preserve_distribution=True
            )
        
        # Geographic generalization
        if 'pat_sarea' in df.columns:
            logger.info(f"Generalizing geography to {self.config.geo_generalization_level}")
            df['pat_sarea'] = self.geo_generalizer.generalize_series(
                df['pat_sarea'],
                target_level=self.config.geo_generalization_level
            )
        
        # Temporal generalization for visit times
        if 'vst_tm' in df.columns:
            logger.info(f"Generalizing time to {self.config.time_generalization_unit}")
            df['vst_tm'] = df['vst_tm'].apply(
                lambda x: self.temporal_generalizer.round_time(
                    x, self.config.time_generalization_unit
                )
            )
        
        return df
    
    def _enforce_k_anonymity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce k-anonymity through suppression or generalization"""
        quasi_identifiers = [
            'pat_age', 'pat_sex', 'pat_sarea',
            'vst_dt', 'vst_tm', 'ktas_lv'
        ]
        
        # Filter to existing columns
        qi = [col for col in quasi_identifiers if col in df.columns]
        
        # Enforce k-anonymity
        df_anonymized, stats = self.k_enforcer.enforce(
            df, qi, method='suppress'
        )
        
        logger.info(f"K-anonymity enforcement stats: {stats}")
        
        # Check if suppression rate is acceptable
        if stats['suppression_rate'] > self.config.max_suppression_rate:
            logger.warning(f"High suppression rate: {stats['suppression_rate']:.2%}")
            logger.info("Attempting generalization instead")
            
            # Try generalization approach
            df_anonymized, stats = self.k_enforcer.enforce(
                df, qi, method='generalize'
            )
            logger.info(f"Generalization stats: {stats}")
        
        return df_anonymized
    
    def _apply_differential_privacy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply differential privacy to numeric columns"""
        # Define column configurations for DP
        column_configs = {}
        
        # Vital signs columns
        vitals = ['sbp', 'dbp', 'pr', 'rr', 'bt', 'spo2']
        for col in vitals:
            if col in df.columns:
                # Determine reasonable bounds
                if col == 'sbp':
                    column_configs[col] = {'sensitivity': 10, 'lower': 60, 'upper': 200}
                elif col == 'dbp':
                    column_configs[col] = {'sensitivity': 10, 'lower': 40, 'upper': 120}
                elif col == 'pr':
                    column_configs[col] = {'sensitivity': 5, 'lower': 40, 'upper': 180}
                elif col == 'rr':
                    column_configs[col] = {'sensitivity': 2, 'lower': 8, 'upper': 40}
                elif col == 'bt':
                    column_configs[col] = {'sensitivity': 0.5, 'lower': 35, 'upper': 41}
                elif col == 'spo2':
                    column_configs[col] = {'sensitivity': 2, 'lower': 70, 'upper': 100}
        
        # Age column
        if 'pat_age' in df.columns:
            column_configs['pat_age'] = {'sensitivity': 5, 'lower': 0, 'upper': 120}
        
        # Apply differential privacy
        if column_configs and self.privacy_accountant.consume(
            self.config.epsilon, "differential_privacy"
        ):
            df = self.dp.apply_to_dataframe(df, column_configs)
            logger.info(f"Applied differential privacy to {len(column_configs)} columns")
        
        return df
    
    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-processing and cleanup"""
        # Ensure data types
        integer_cols = ['pat_age', 'ktas_lv', 'sbp', 'dbp', 'pr', 'rr', 'spo2']
        for col in integer_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
                df[col] = df[col].round().astype(int)
        
        # Sort by visit date/time if available
        if 'vst_dt' in df.columns:
            df = df.sort_values('vst_dt')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def generate_with_constraints(self, n_patients: int,
                                 constraints: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate synthetic data with specific constraints
        
        Args:
            n_patients: Number of patients
            constraints: Dictionary of constraints
                        e.g., {'min_k': 10, 'max_risk': 0.1}
                        
        Returns:
            Synthetic data satisfying constraints
        """
        # Update config based on constraints
        if 'min_k' in constraints:
            self.config.k_threshold = constraints['min_k']
        if 'max_risk' in constraints:
            self.config.max_risk_score = constraints['max_risk']
        if 'epsilon' in constraints:
            self.config.epsilon = constraints['epsilon']
        
        # Generate with iterative refinement
        max_attempts = 5
        for attempt in range(max_attempts):
            logger.info(f"Generation attempt {attempt + 1}/{max_attempts}")
            
            # Generate data
            synthetic_df, validation = self.generate(n_patients, validate_privacy=True)
            
            # Check if constraints are met
            if validation and self._check_constraints(validation, constraints):
                logger.info("All constraints satisfied")
                return synthetic_df
            
            # Adjust parameters for next attempt
            if validation:
                self._adjust_parameters(validation, constraints)
        
        logger.warning(f"Could not satisfy all constraints after {max_attempts} attempts")
        return synthetic_df
    
    def _check_constraints(self, validation: PrivacyValidationResult,
                          constraints: Dict[str, Any]) -> bool:
        """Check if validation results meet constraints"""
        if 'min_k' in constraints:
            if validation.overall_metrics.k_anonymity < constraints['min_k']:
                return False
        
        if 'max_risk' in constraints:
            if validation.overall_metrics.risk_score > constraints['max_risk']:
                return False
        
        if 'min_l' in constraints:
            if validation.overall_metrics.l_diversity < constraints['min_l']:
                return False
        
        return True
    
    def _adjust_parameters(self, validation: PrivacyValidationResult,
                          constraints: Dict[str, Any]):
        """Adjust generation parameters based on validation results"""
        # If k-anonymity is too low, increase generalization
        if validation.overall_metrics.k_anonymity < self.config.k_threshold:
            self.config.age_group_size = min(self.config.age_group_size + 5, 20)
            logger.info(f"Increased age group size to {self.config.age_group_size}")
        
        # If risk is too high, increase privacy budget
        if validation.overall_metrics.risk_score > self.config.max_risk_score:
            self.config.epsilon = max(self.config.epsilon * 0.8, 0.1)
            logger.info(f"Decreased epsilon to {self.config.epsilon}")
    
    def save_results(self, synthetic_df: pd.DataFrame,
                    validation: Optional[PrivacyValidationResult],
                    output_dir: str):
        """
        Save generation results and reports
        
        Args:
            synthetic_df: Generated synthetic data
            validation: Privacy validation results
            output_dir: Directory to save outputs
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save synthetic data
        data_path = os.path.join(output_dir, 'synthetic_data.parquet')
        synthetic_df.to_parquet(data_path, index=False)
        logger.info(f"Saved synthetic data to {data_path}")
        
        # Save validation report
        if validation:
            report_path = os.path.join(output_dir, 'privacy_validation_report.html')
            self.privacy_validator.generate_report(validation, report_path)
            
            # Save validation metrics as JSON
            metrics_path = os.path.join(output_dir, 'privacy_metrics.json')
            metrics = {
                'k_anonymity': validation.overall_metrics.k_anonymity,
                'l_diversity': validation.overall_metrics.l_diversity,
                't_closeness': validation.overall_metrics.t_closeness,
                'risk_score': validation.overall_metrics.risk_score,
                'risk_level': validation.overall_metrics.risk_level,
                'validation_passed': validation.validation_passed,
                'timestamp': validation.timestamp.isoformat()
            }
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved privacy metrics to {metrics_path}")
        
        # Save configuration
        config_path = os.path.join(output_dir, 'generation_config.json')
        config_dict = {
            'k_threshold': self.config.k_threshold,
            'l_threshold': self.config.l_threshold,
            'epsilon': self.config.epsilon,
            'delta': self.config.delta,
            'age_group_size': self.config.age_group_size,
            'geo_generalization_level': self.config.geo_generalization_level,
            'time_generalization_unit': self.config.time_generalization_unit,
            'max_risk_score': self.config.max_risk_score
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Saved configuration to {config_path}")


def main():
    """Example usage of enhanced generator"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate privacy-enhanced synthetic NEDIS data')
    parser.add_argument('--db', type=str, default='data/nedis_data.duckdb',
                       help='Path to database')
    parser.add_argument('--n-patients', type=int, default=1000,
                       help='Number of patients to generate')
    parser.add_argument('--k-threshold', type=int, default=5,
                       help='K-anonymity threshold')
    parser.add_argument('--epsilon', type=float, default=1.0,
                       help='Differential privacy epsilon')
    parser.add_argument('--output-dir', type=str, default='outputs/enhanced',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create privacy config
    config = PrivacyConfig(
        k_threshold=args.k_threshold,
        epsilon=args.epsilon
    )
    
    # Initialize generator
    generator = EnhancedSyntheticGenerator(args.db, config)
    
    # Generate synthetic data
    synthetic_df, validation = generator.generate(
        args.n_patients,
        start_date=datetime(2017, 1, 1),
        end_date=datetime(2017, 12, 31)
    )
    
    # Save results
    generator.save_results(synthetic_df, validation, args.output_dir)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Enhanced Synthetic Data Generation Complete")
    print(f"{'='*60}")
    print(f"Generated records: {len(synthetic_df)}")
    if validation:
        print(f"Risk level: {validation.overall_metrics.risk_level}")
        print(f"K-anonymity: {validation.overall_metrics.k_anonymity}")
        print(f"L-diversity: {validation.overall_metrics.l_diversity:.1f}")
        print(f"Risk score: {validation.overall_metrics.risk_score:.2%}")
        print(f"Validation passed: {validation.validation_passed}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()