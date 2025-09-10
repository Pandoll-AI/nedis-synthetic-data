"""
K-Anonymity Implementation

Validates and enforces k-anonymity for privacy protection.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class KAnonymityResult:
    """Results from k-anonymity validation"""
    k_value: int
    min_group_size: int
    max_group_size: int
    mean_group_size: float
    num_groups: int
    num_violations: int
    violation_records: List[int]
    satisfied: bool
    groups_distribution: Dict[int, int]  # group_size -> count


class KAnonymityValidator:
    """
    Validates k-anonymity for a dataset
    """
    
    def __init__(self, k_threshold: int = 5):
        """
        Initialize k-anonymity validator
        
        Args:
            k_threshold: Minimum k value required
        """
        self.k_threshold = k_threshold
    
    def validate(self, df: pd.DataFrame, 
                quasi_identifiers: List[str]) -> KAnonymityResult:
        """
        Validate k-anonymity for given quasi-identifiers
        
        Args:
            df: DataFrame to validate
            quasi_identifiers: List of quasi-identifier columns
            
        Returns:
            KAnonymityResult with validation details
        """
        # Filter to valid quasi-identifiers
        valid_qi = [qi for qi in quasi_identifiers if qi in df.columns]
        
        if not valid_qi:
            logger.warning("No valid quasi-identifiers found")
            return KAnonymityResult(
                k_value=len(df),
                min_group_size=len(df),
                max_group_size=len(df),
                mean_group_size=len(df),
                num_groups=1,
                num_violations=0,
                violation_records=[],
                satisfied=True,
                groups_distribution={len(df): 1}
            )
        
        # Create equivalence classes
        groups = df.groupby(valid_qi).size().reset_index(name='group_size')
        
        # Calculate statistics
        min_group_size = groups['group_size'].min()
        max_group_size = groups['group_size'].max()
        mean_group_size = groups['group_size'].mean()
        num_groups = len(groups)
        
        # Find violations
        violations = groups[groups['group_size'] < self.k_threshold]
        num_violations = len(violations)
        
        # Get indices of violation records
        violation_records = []
        if num_violations > 0:
            for _, violation_group in violations.iterrows():
                # Build query to find matching records
                query_parts = []
                for qi in valid_qi:
                    val = violation_group[qi]
                    if pd.isna(val):
                        query_parts.append(f"{qi}.isna()")
                    else:
                        query_parts.append(f"{qi} == @violation_group['{qi}']")
                
                query = " & ".join(query_parts)
                matching_indices = df.query(query).index.tolist()
                violation_records.extend(matching_indices)
        
        # Group size distribution
        groups_distribution = groups['group_size'].value_counts().to_dict()
        
        # Determine k-value (minimum group size)
        k_value = min_group_size
        
        result = KAnonymityResult(
            k_value=k_value,
            min_group_size=min_group_size,
            max_group_size=max_group_size,
            mean_group_size=mean_group_size,
            num_groups=num_groups,
            num_violations=num_violations,
            violation_records=violation_records,
            satisfied=(k_value >= self.k_threshold),
            groups_distribution=groups_distribution
        )
        
        logger.info(f"K-anonymity validation: k={k_value}, satisfied={result.satisfied}")
        logger.info(f"Groups: {num_groups}, Violations: {num_violations}")
        
        return result
    
    def find_optimal_generalization(self, df: pd.DataFrame,
                                   quasi_identifiers: List[str],
                                   generalization_hierarchies: Dict[str, List]) -> Dict[str, int]:
        """
        Find optimal generalization levels to achieve k-anonymity
        
        Args:
            df: DataFrame to analyze
            quasi_identifiers: List of quasi-identifier columns
            generalization_hierarchies: Dict mapping QI to list of generalization levels
            
        Returns:
            Dict mapping QI to optimal generalization level
        """
        # Start with no generalization
        current_levels = {qi: 0 for qi in quasi_identifiers}
        
        # Iteratively increase generalization until k-anonymity is satisfied
        max_iterations = 10
        for iteration in range(max_iterations):
            # Apply current generalization levels
            generalized_df = self._apply_generalization(
                df, current_levels, generalization_hierarchies
            )
            
            # Check k-anonymity
            result = self.validate(generalized_df, quasi_identifiers)
            
            if result.satisfied:
                logger.info(f"K-anonymity achieved with levels: {current_levels}")
                return current_levels
            
            # Find QI with most violations and increase its generalization
            qi_violations = self._count_qi_violations(
                generalized_df, quasi_identifiers, result.violation_records
            )
            
            if not qi_violations:
                break
            
            # Increase generalization for QI with most violations
            worst_qi = max(qi_violations, key=qi_violations.get)
            max_level = len(generalization_hierarchies.get(worst_qi, [])) - 1
            
            if current_levels[worst_qi] < max_level:
                current_levels[worst_qi] += 1
                logger.info(f"Increasing generalization for {worst_qi} to level {current_levels[worst_qi]}")
            else:
                # Try next worst QI
                for qi in sorted(qi_violations, key=qi_violations.get, reverse=True):
                    max_level = len(generalization_hierarchies.get(qi, [])) - 1
                    if current_levels[qi] < max_level:
                        current_levels[qi] += 1
                        logger.info(f"Increasing generalization for {qi} to level {current_levels[qi]}")
                        break
                else:
                    logger.warning("Cannot achieve k-anonymity with available generalizations")
                    break
        
        return current_levels
    
    def _apply_generalization(self, df: pd.DataFrame,
                             levels: Dict[str, int],
                             hierarchies: Dict[str, List]) -> pd.DataFrame:
        """Apply generalization levels to dataframe"""
        result_df = df.copy()
        
        for qi, level in levels.items():
            if qi in hierarchies and level < len(hierarchies[qi]):
                gen_func = hierarchies[qi][level]
                if callable(gen_func):
                    result_df[qi] = result_df[qi].apply(gen_func)
        
        return result_df
    
    def _count_qi_violations(self, df: pd.DataFrame,
                           quasi_identifiers: List[str],
                           violation_indices: List[int]) -> Dict[str, int]:
        """Count violations per quasi-identifier"""
        if not violation_indices:
            return {}
        
        violation_df = df.loc[violation_indices]
        qi_violations = {}
        
        for qi in quasi_identifiers:
            if qi in violation_df.columns:
                # Count unique values in violations
                unique_count = violation_df[qi].nunique()
                qi_violations[qi] = unique_count
        
        return qi_violations


class KAnonymityEnforcer:
    """
    Enforces k-anonymity through suppression and generalization
    """
    
    def __init__(self, k_threshold: int = 5, max_suppression_rate: float = 0.05):
        """
        Initialize k-anonymity enforcer
        
        Args:
            k_threshold: Minimum k value to enforce
            max_suppression_rate: Maximum fraction of records to suppress
        """
        self.k_threshold = k_threshold
        self.max_suppression_rate = max_suppression_rate
        self.validator = KAnonymityValidator(k_threshold)
    
    def enforce(self, df: pd.DataFrame,
               quasi_identifiers: List[str],
               method: str = 'suppress') -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Enforce k-anonymity on dataset
        
        Args:
            df: DataFrame to process
            quasi_identifiers: List of quasi-identifier columns
            method: 'suppress' to remove records, 'generalize' to generalize values
            
        Returns:
            Tuple of (processed DataFrame, enforcement statistics)
        """
        logger.info(f"Enforcing k-anonymity with k={self.k_threshold}, method={method}")
        
        if method == 'suppress':
            return self._enforce_by_suppression(df, quasi_identifiers)
        elif method == 'generalize':
            return self._enforce_by_generalization(df, quasi_identifiers)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _enforce_by_suppression(self, df: pd.DataFrame,
                               quasi_identifiers: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """
        Enforce k-anonymity by suppressing violating records
        """
        original_size = len(df)
        result_df = df.copy()
        
        # Iteratively remove violating records
        max_iterations = 10
        total_suppressed = 0
        
        for iteration in range(max_iterations):
            # Validate current state
            validation = self.validator.validate(result_df, quasi_identifiers)
            
            if validation.satisfied:
                break
            
            # Remove violating records
            if validation.violation_records:
                records_to_remove = validation.violation_records
                result_df = result_df.drop(index=records_to_remove, errors='ignore')
                result_df = result_df.reset_index(drop=True)
                total_suppressed += len(records_to_remove)
                
                logger.info(f"Iteration {iteration + 1}: Suppressed {len(records_to_remove)} records")
                
                # Check suppression limit
                suppression_rate = total_suppressed / original_size
                if suppression_rate > self.max_suppression_rate:
                    logger.warning(f"Suppression rate {suppression_rate:.2%} exceeds limit")
                    break
        
        # Final validation
        final_validation = self.validator.validate(result_df, quasi_identifiers)
        
        stats = {
            'original_size': original_size,
            'final_size': len(result_df),
            'suppressed_count': total_suppressed,
            'suppression_rate': total_suppressed / original_size,
            'k_achieved': final_validation.k_value,
            'satisfied': final_validation.satisfied,
            'num_groups': final_validation.num_groups
        }
        
        logger.info(f"Suppression complete: {stats['suppressed_count']} records removed "
                   f"({stats['suppression_rate']:.2%}), k={stats['k_achieved']}")
        
        return result_df, stats
    
    def _enforce_by_generalization(self, df: pd.DataFrame,
                                  quasi_identifiers: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """
        Enforce k-anonymity by generalizing quasi-identifiers
        """
        from .generalization import AgeGeneralizer, GeographicGeneralizer, TemporalGeneralizer
        
        result_df = df.copy()
        
        # Define generalization strategies
        age_gen = AgeGeneralizer(group_size=5)  # Start with 5-year groups
        geo_gen = GeographicGeneralizer()
        time_gen = TemporalGeneralizer()
        
        # Apply progressive generalization
        generalization_levels = {qi: 0 for qi in quasi_identifiers}
        
        for level in range(5):  # Max 5 levels of generalization
            # Apply generalization based on column type
            for qi in quasi_identifiers:
                if 'age' in qi.lower():
                    if level == 0:
                        result_df[qi] = age_gen.generalize_series(result_df[qi], method='random')
                    elif level == 1:
                        age_gen.group_size = 10
                        result_df[qi] = age_gen.generalize_series(result_df[qi], method='random')
                    else:
                        age_gen.group_size = 20
                        result_df[qi] = age_gen.generalize_series(result_df[qi], method='center')
                
                elif 'area' in qi.lower() or 'region' in qi.lower():
                    if level == 0:
                        result_df[qi] = geo_gen.generalize_series(result_df[qi], 'district')
                    else:
                        result_df[qi] = geo_gen.generalize_series(result_df[qi], 'province')
                
                elif 'tm' in qi.lower() or 'time' in qi.lower():
                    if level == 0:
                        result_df[qi] = result_df[qi].apply(
                            lambda x: time_gen.round_time(x, 'hour')
                        )
                    elif level == 1:
                        result_df[qi] = result_df[qi].apply(
                            lambda x: time_gen.round_time(x, 'shift')
                        )
                    else:
                        result_df[qi] = result_df[qi].apply(
                            lambda x: time_gen.round_time(x, 'day')
                        )
                
                generalization_levels[qi] = level
            
            # Check if k-anonymity is satisfied
            validation = self.validator.validate(result_df, quasi_identifiers)
            if validation.satisfied:
                break
        
        # If still not satisfied, apply suppression to remaining violations
        if not validation.satisfied and validation.violation_records:
            result_df = result_df.drop(index=validation.violation_records, errors='ignore')
            result_df = result_df.reset_index(drop=True)
            
            # Re-validate
            validation = self.validator.validate(result_df, quasi_identifiers)
        
        stats = {
            'original_size': len(df),
            'final_size': len(result_df),
            'generalization_levels': generalization_levels,
            'suppressed_count': len(df) - len(result_df),
            'k_achieved': validation.k_value,
            'satisfied': validation.satisfied,
            'num_groups': validation.num_groups
        }
        
        logger.info(f"Generalization complete: levels={generalization_levels}, "
                   f"k={stats['k_achieved']}, satisfied={stats['satisfied']}")
        
        return result_df, stats