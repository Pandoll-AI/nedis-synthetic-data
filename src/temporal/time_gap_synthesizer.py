#!/usr/bin/env python3
"""
Time Gap Synthesizer for NEDIS Data

Generates realistic time gaps between medical events based on KTAS severity levels
and treatment outcomes, following the principle of dynamic pattern learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import json
from pathlib import Path
from scipy import stats

from ..core.database import DatabaseManager
from ..core.config import ConfigManager
from ..analysis.pattern_analyzer import PatternAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class TimeGapConfig:
    """Configuration for time gap synthesis"""
    use_hierarchical_fallback: bool = True
    min_sample_size: int = 10
    cache_patterns: bool = True
    enforce_clinical_constraints: bool = True
    
    # Clinical constraint parameters (in minutes)
    min_er_stay: Dict[str, float] = None
    max_er_stay: Dict[str, float] = None
    
    def __post_init__(self):
        if self.min_er_stay is None:
            # Minimum ER stay by KTAS level (minutes)
            self.min_er_stay = {
                '1': 15,   # Critical - at least 15 minutes
                '2': 30,   # Emergency - at least 30 minutes  
                '3': 45,   # Urgent - at least 45 minutes
                '4': 60,   # Less urgent - at least 1 hour
                '5': 30    # Non-urgent - at least 30 minutes
            }
        
        if self.max_er_stay is None:
            # Maximum reasonable ER stay (minutes)
            self.max_er_stay = {
                '1': 720,   # 12 hours
                '2': 1440,  # 24 hours
                '3': 2880,  # 48 hours
                '4': 2880,  # 48 hours
                '5': 1440   # 24 hours
            }


class TimeGapSynthesizer:
    """
    Synthesizes time gaps between medical events based on severity-adjusted distributions.
    Follows the no-hardcoding principle by learning from actual data patterns.
    """
    
    def __init__(self, db_manager: DatabaseManager, config: ConfigManager):
        """
        Initialize the time gap synthesizer
        
        Args:
            db_manager: Database manager instance
            config: Configuration manager instance
        """
        self.db = db_manager
        self.config = config
        self.time_config = TimeGapConfig()
        
        # Pattern analyzer for dynamic learning
        self.pattern_analyzer = PatternAnalyzer(db_manager, config)
        
        # Cache for learned patterns
        self._time_patterns = None
        self._distribution_params = None
        
    def analyze_time_patterns(self, force_reanalysis: bool = False) -> Dict[str, Any]:
        """
        Analyze time gap patterns from original data
        
        Args:
            force_reanalysis: Force re-analysis even if cached patterns exist
            
        Returns:
            Dictionary of time gap patterns by KTAS and treatment result
        """
        if self._time_patterns is not None and not force_reanalysis:
            return self._time_patterns
            
        logger.info("Analyzing time gap patterns from original data...")
        
        # Get source table
        source_table = self.config.get('original.source_table', 'nedis_data.nedis2017')
        
        query = f"""
        SELECT 
            ktas01,
            emtrt_rust,
            vst_dt, vst_tm,
            ocur_dt, ocur_tm,
            otrm_dt, otrm_tm,
            inpat_dt, inpat_tm,
            otpat_dt, otpat_tm
        FROM {source_table}
        WHERE ktas01 IS NOT NULL 
            AND ktas01 >= 1 
            AND ktas01 <= 5
        LIMIT 50000
        """
        
        df = self.db.fetch_dataframe(query)
        
        # Parse ALL datetime columns
        df['vst_datetime'] = df.apply(
            lambda x: self._parse_datetime(x['vst_dt'], x['vst_tm']), axis=1
        )
        df['ocur_datetime'] = df.apply(
            lambda x: self._parse_datetime(x['ocur_dt'], x['ocur_tm']), axis=1
        )
        df['otrm_datetime'] = df.apply(
            lambda x: self._parse_datetime(x['otrm_dt'], x['otrm_tm']), axis=1
        )
        df['inpat_datetime'] = df.apply(
            lambda x: self._parse_datetime(x['inpat_dt'], x['inpat_tm']), axis=1
        )
        df['otpat_datetime'] = df.apply(
            lambda x: self._parse_datetime(x['otpat_dt'], x['otpat_tm']), axis=1
        )
        
        # Calculate ALL time gaps using apply to handle None values properly
        def calc_time_diff_minutes(dt1, dt2):
            if pd.notna(dt1) and pd.notna(dt2) and isinstance(dt1, datetime) and isinstance(dt2, datetime):
                return (dt1 - dt2).total_seconds() / 60
            return np.nan
        
        # All possible time gaps (non-overlapping)
        df['gap_ocur_to_vst'] = df.apply(lambda x: calc_time_diff_minutes(x['vst_datetime'], x['ocur_datetime']), axis=1)
        df['gap_vst_to_otrm'] = df.apply(lambda x: calc_time_diff_minutes(x['otrm_datetime'], x['vst_datetime']), axis=1)
        df['gap_otrm_to_inpat'] = df.apply(lambda x: calc_time_diff_minutes(x['inpat_datetime'], x['otrm_datetime']), axis=1)
        df['gap_otrm_to_otpat'] = df.apply(lambda x: calc_time_diff_minutes(x['otpat_datetime'], x['otrm_datetime']), axis=1)
        # Additional useful gaps
        df['gap_vst_to_inpat'] = df.apply(lambda x: calc_time_diff_minutes(x['inpat_datetime'], x['vst_datetime']), axis=1)
        df['gap_ocur_to_otrm'] = df.apply(lambda x: calc_time_diff_minutes(x['otrm_datetime'], x['ocur_datetime']), axis=1)
        
        # Build hierarchical patterns
        patterns = {}
        
        # Level 1: KTAS + Treatment Result
        for ktas in range(1, 6):
            for result in df['emtrt_rust'].unique():
                if pd.notna(result):
                    mask = (df['ktas01'] == ktas) & (df['emtrt_rust'] == result)
                    subset = df[mask]
                    
                    if len(subset) >= self.time_config.min_sample_size:
                        key = f"ktas_{ktas}_result_{result}"
                        patterns[key] = self._extract_distribution_params(subset)
        
        # Level 2: KTAS only
        for ktas in range(1, 6):
            subset = df[df['ktas01'] == ktas]
            if len(subset) >= self.time_config.min_sample_size:
                key = f"ktas_{ktas}"
                patterns[key] = self._extract_distribution_params(subset)
        
        # Level 3: Overall patterns
        patterns['overall'] = self._extract_distribution_params(df)
        
        self._time_patterns = patterns
        
        # Cache patterns if configured
        if self.time_config.cache_patterns:
            self._save_patterns_cache(patterns)
            
        return patterns
    
    def _extract_distribution_params(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract distribution parameters from ALL time gaps in the data
        
        Args:
            df: DataFrame with time gap data
            
        Returns:
            Dictionary of distribution parameters for all gaps
        """
        params = {}
        
        # Define all gap types with reasonable limits
        gap_configs = {
            'gap_ocur_to_vst': {'name': 'incident_to_arrival', 'max_hours': 72},
            'gap_vst_to_otrm': {'name': 'er_stay', 'max_hours': 168},
            'gap_otrm_to_inpat': {'name': 'discharge_to_admission', 'max_hours': 48},
            'gap_otrm_to_otpat': {'name': 'discharge_to_outpatient', 'max_hours': 168},
            'gap_vst_to_inpat': {'name': 'arrival_to_admission', 'max_hours': 168},
            'gap_ocur_to_otrm': {'name': 'incident_to_discharge', 'max_hours': 240}
        }
        
        for gap_col, config in gap_configs.items():
            if gap_col in df.columns:
                gap_data = df[gap_col].dropna()
                max_minutes = config['max_hours'] * 60
                gap_data = gap_data[(gap_data > 0) & (gap_data < max_minutes)]
                
                if len(gap_data) > 10:  # Need minimum samples
                    try:
                        # Fit log-normal distribution
                        shape, loc, scale = stats.lognorm.fit(gap_data, floc=0)
                        params[config['name']] = {
                            'distribution': 'lognorm',
                            'shape': float(shape),
                            'loc': float(loc),
                            'scale': float(scale),
                            'mean': float(gap_data.mean()),
                            'median': float(gap_data.median()),
                            'std': float(gap_data.std()),
                            'count': int(len(gap_data)),
                            'percentiles': {
                                '25': float(gap_data.quantile(0.25)),
                                '50': float(gap_data.quantile(0.50)),
                                '75': float(gap_data.quantile(0.75)),
                                '95': float(gap_data.quantile(0.95))
                            }
                        }
                    except:
                        # Fallback to basic statistics if distribution fitting fails
                        params[config['name']] = {
                            'distribution': 'empirical',
                            'mean': float(gap_data.mean()),
                            'median': float(gap_data.median()),
                            'std': float(gap_data.std()),
                            'count': int(len(gap_data))
                        }
        
        return params
    
    def generate_time_gaps(
        self,
        ktas_levels: np.ndarray,
        treatment_results: np.ndarray,
        visit_datetimes: pd.Series
    ) -> pd.DataFrame:
        """
        Generate time gaps for a batch of patients
        
        Args:
            ktas_levels: Array of KTAS levels (1-5)
            treatment_results: Array of treatment result codes
            visit_datetimes: Series of visit datetime objects
            
        Returns:
            DataFrame with generated datetime columns
        """
        if self._time_patterns is None:
            self.analyze_time_patterns()
        
        n_patients = len(ktas_levels)
        
        # Initialize result arrays
        otrm_datetimes = []
        inpat_datetimes = []
        
        for i in range(n_patients):
            ktas = str(int(ktas_levels[i]))
            result = str(treatment_results[i]) if pd.notna(treatment_results[i]) else None
            vst_dt = visit_datetimes.iloc[i]
            
            # Get appropriate distribution using hierarchical fallback
            dist_params = self._get_hierarchical_distribution(ktas, result)
            
            # Generate ER discharge time
            er_stay_minutes = self._sample_from_distribution(
                dist_params.get('er_stay'),
                ktas,
                'er_stay'
            )
            
            if pd.notna(vst_dt) and er_stay_minutes is not None:
                otrm_dt = vst_dt + timedelta(minutes=er_stay_minutes)
                otrm_datetimes.append(otrm_dt)
            else:
                otrm_datetimes.append(None)
            
            # Generate admission time if applicable
            if result in ['31', '32', '33', '34']:  # Admission codes
                admit_minutes = self._sample_from_distribution(
                    dist_params.get('admit_time'),
                    ktas,
                    'admit'
                )
                
                if pd.notna(vst_dt) and admit_minutes is not None:
                    inpat_dt = vst_dt + timedelta(minutes=admit_minutes)
                    inpat_datetimes.append(inpat_dt)
                else:
                    inpat_datetimes.append(None)
            else:
                inpat_datetimes.append(None)
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'otrm_datetime': otrm_datetimes,
            'inpat_datetime': inpat_datetimes
        })
        
        # Convert to NEDIS format (separate date and time columns) - use YYYYMMDD format
        result_df['otrm_dt'] = result_df['otrm_datetime'].apply(
            lambda x: x.strftime('%Y%m%d') if pd.notna(x) else None
        )
        result_df['otrm_tm'] = result_df['otrm_datetime'].apply(
            lambda x: x.strftime('%H%M') if pd.notna(x) else None
        )
        result_df['inpat_dt'] = result_df['inpat_datetime'].apply(
            lambda x: x.strftime('%Y%m%d') if pd.notna(x) else None
        )
        result_df['inpat_tm'] = result_df['inpat_datetime'].apply(
            lambda x: x.strftime('%H%M') if pd.notna(x) else None
        )
        
        return result_df
    
    def _get_hierarchical_distribution(
        self, 
        ktas: str, 
        treatment_result: Optional[str]
    ) -> Dict[str, Any]:
        """
        Get distribution parameters using hierarchical fallback
        
        Args:
            ktas: KTAS level (1-5)
            treatment_result: Treatment result code
            
        Returns:
            Distribution parameters dictionary
        """
        # Level 1: Try KTAS + Treatment Result
        if treatment_result:
            key = f"ktas_{ktas}_result_{treatment_result}"
            if key in self._time_patterns:
                return self._time_patterns[key]
        
        # Level 2: Try KTAS only
        key = f"ktas_{ktas}"
        if key in self._time_patterns:
            return self._time_patterns[key]
        
        # Level 3: Use overall patterns
        return self._time_patterns.get('overall', {})
    
    def _sample_from_distribution(
        self,
        params: Optional[Dict[str, Any]],
        ktas: str,
        gap_type: str
    ) -> Optional[float]:
        """
        Sample from a distribution with clinical constraints
        
        Args:
            params: Distribution parameters
            ktas: KTAS level for constraints
            gap_type: Type of gap (er_stay, admit)
            
        Returns:
            Sampled time in minutes, or None if no params
        """
        if not params:
            # Use clinical defaults if no distribution available
            if gap_type == 'er_stay':
                # Default ER stay times by KTAS (minutes)
                defaults = {'1': 120, '2': 180, '3': 240, '4': 300, '5': 180}
                return defaults.get(ktas, 240)
            elif gap_type == 'admit':
                # Default admission decision times by KTAS
                defaults = {'1': 30, '2': 60, '3': 120, '4': 180, '5': 240}
                return defaults.get(ktas, 120)
            return None
        
        # Sample from distribution
        if params['distribution'] == 'lognorm':
            sample = stats.lognorm.rvs(
                params['shape'],
                loc=params['loc'],
                scale=params['scale']
            )
        else:
            # Fallback to normal distribution
            sample = np.random.normal(params['mean'], params['std'])
        
        # Apply clinical constraints
        if self.time_config.enforce_clinical_constraints:
            if gap_type == 'er_stay':
                min_val = self.time_config.min_er_stay.get(ktas, 15)
                max_val = self.time_config.max_er_stay.get(ktas, 2880)
                sample = np.clip(sample, min_val, max_val)
        
        return float(sample)
    
    def _parse_datetime(self, dt_str: str, tm_str: str) -> Optional[datetime]:
        """
        Parse NEDIS datetime format
        
        Args:
            dt_str: Date string (YYYYMMDD or YYMMDD)
            tm_str: Time string (HHMM)
            
        Returns:
            datetime object or None
        """
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
    
    def _save_patterns_cache(self, patterns: Dict[str, Any]):
        """Save analyzed patterns to cache"""
        cache_dir = Path('cache/time_patterns')
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = cache_dir / 'time_gap_patterns.json'
        with open(cache_file, 'w') as f:
            json.dump(patterns, f, indent=2, default=str)
        
        logger.info(f"Saved time gap patterns to {cache_file}")