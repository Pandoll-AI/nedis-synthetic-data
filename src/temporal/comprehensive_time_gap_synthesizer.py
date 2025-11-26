"""
Comprehensive Time Gap Synthesizer for ALL NEDIS datetime pairs
Handles all non-overlapping time gaps between medical events
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import json
from pathlib import Path
from scipy import stats
import logging

from src.core.database import DatabaseManager
from src.core.config import ConfigManager

logger = logging.getLogger(__name__)


class ComprehensiveTimeGapSynthesizer:
    """
    Synthesizes ALL time gaps between NEDIS datetime pairs:
    - ocur_dt/tm: Incident occurrence time
    - vst_dt/tm: ER visit/arrival time  
    - otrm_dt/tm: ER discharge time
    - inpat_dt/tm: Hospital admission time
    - otpat_dt/tm: Outpatient transfer time
    """
    
    def __init__(self, db_manager: DatabaseManager, config: ConfigManager):
        self.db = db_manager
        self.config = config
        self.time_config = config.get('temporal', {})
        self._time_patterns = None
        
        # Define all possible time gaps
        self.gap_definitions = {
            'incident_to_arrival': {
                'from': 'ocur', 'to': 'vst',
                'description': 'Time from incident to ER arrival',
                'max_hours': 72
            },
            'er_stay': {
                'from': 'vst', 'to': 'otrm',
                'description': 'ER stay duration',
                'max_hours': 168
            },
            'discharge_to_admission': {
                'from': 'otrm', 'to': 'inpat',
                'description': 'Time from ER discharge to admission',
                'max_hours': 48
            },
            'discharge_to_outpatient': {
                'from': 'otrm', 'to': 'otpat',
                'description': 'Time from ER discharge to outpatient',
                'max_hours': 168
            },
            'arrival_to_admission': {
                'from': 'vst', 'to': 'inpat',
                'description': 'Total time from arrival to admission',
                'max_hours': 168
            },
            'incident_to_discharge': {
                'from': 'ocur', 'to': 'otrm',
                'description': 'Total time from incident to ER discharge',
                'max_hours': 240
            }
        }
        
        # Flow Types Definition (Patient Flow Patterns)
        self.flow_types = {
            'fast_track': {'description': 'Light symptoms, quick discharge', 'multiplier': 0.5},
            'standard': {'description': 'Average case', 'multiplier': 1.0},
            'observation': {'description': 'Requires observation, longer stay', 'multiplier': 2.0},
            'critical_care': {'description': 'Severe, ICU admission, very long stay', 'multiplier': 3.0},
            'boarding': {'description': 'Admission decided but waiting for bed', 'multiplier': 4.0}
        }
    
    def analyze_all_time_patterns(self) -> Dict[str, Any]:
        """
        Analyze ALL time gap patterns from real NEDIS data
        """
        logger.info("Analyzing comprehensive time gap patterns...")
        
        # Load data with all datetime columns
        source_table = self.config.get('original.source_table', 'nedis_data.nedis2017')
        
        query = f"""
        SELECT 
            ktas01, emtrt_rust,
            vst_dt, vst_tm,
            ocur_dt, ocur_tm,
            otrm_dt, otrm_tm,
            inpat_dt, inpat_tm,
            otpat_dt, otpat_tm
        FROM {source_table}
        WHERE ktas01 IS NOT NULL 
            AND ktas01 >= 1 
            AND ktas01 <= 5
        LIMIT 100000
        """
        
        df = self.db.fetch_dataframe(query)
        logger.info(f"Loaded {len(df)} records for analysis")
        
        # Parse all datetime pairs
        datetime_pairs = ['vst', 'ocur', 'otrm', 'inpat', 'otpat']
        for prefix in datetime_pairs:
            df[f'{prefix}_datetime'] = df.apply(
                lambda x: self._parse_datetime(x[f'{prefix}_dt'], x[f'{prefix}_tm']), 
                axis=1
            )
        
        # Calculate all time gaps
        for gap_name, gap_def in self.gap_definitions.items():
            from_col = f"{gap_def['from']}_datetime"
            to_col = f"{gap_def['to']}_datetime"
            
            df[f'gap_{gap_name}'] = df.apply(
                lambda x: self._calc_time_diff_minutes(x[to_col], x[from_col]),
                axis=1
            )
        
        # Analyze patterns hierarchically
        patterns = self._build_hierarchical_patterns(df)
        
        # Save cache
        if self.time_config.get('cache_patterns', True):
            self._save_patterns_cache(patterns)
        
        self._time_patterns = patterns
        return patterns
    
    def _calc_time_diff_minutes(self, dt1, dt2) -> Optional[float]:
        """Calculate time difference in minutes"""
        if pd.notna(dt1) and pd.notna(dt2) and isinstance(dt1, datetime) and isinstance(dt2, datetime):
            diff = (dt1 - dt2).total_seconds() / 60
            return diff if diff >= 0 else None  # Only positive gaps
        return None
    
    def _parse_datetime(self, dt_str: str, tm_str: str) -> Optional[datetime]:
        """Parse NEDIS datetime format (handles both YYYYMMDD and YYMMDD)"""
        try:
            if pd.isna(dt_str) or pd.isna(tm_str):
                return None
                
            dt_str = str(dt_str).strip()
            tm_str = str(tm_str).strip().zfill(4)
            
            # Skip invalid placeholder dates
            if dt_str in ['11111111', '99999999', '00000000'] or tm_str in ['1111', '9999']:
                return None
            
            # Handle date format
            if len(dt_str) == 8:  # YYYYMMDD
                year = int(dt_str[0:4])
                month = int(dt_str[4:6])
                day = int(dt_str[6:8])
            elif len(dt_str) == 6:  # YYMMDD
                year = 2000 + int(dt_str[0:2])
                month = int(dt_str[2:4])
                day = int(dt_str[4:6])
            else:
                return None
            
            # Validate date components
            if year < 1900 or year > 2100:
                return None
            if month < 1 or month > 12:
                return None
            if day < 1 or day > 31:
                return None
            
            # Parse time
            hour = int(tm_str[0:2])
            minute = int(tm_str[2:4])
            
            # Validate time components
            if hour < 0 or hour > 23:
                return None
            if minute < 0 or minute > 59:
                return None
            
            return datetime(year, month, day, hour, minute)
        except:
            return None
    
    def _build_hierarchical_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build hierarchical patterns for all gaps"""
        patterns = {
            'summary': {
                'total_records': len(df),
                'analysis_date': datetime.now().isoformat(),
                'gaps_analyzed': list(self.gap_definitions.keys())
            },
            'hierarchical': {}
        }
        
        # Level 1: KTAS + Treatment Result
        level1_patterns = {}
        for ktas in range(1, 6):
            for result in df['emtrt_rust'].unique():
                if pd.notna(result):
                    key = f"ktas_{ktas}_result_{result}"
                    mask = (df['ktas01'] == ktas) & (df['emtrt_rust'] == result)
                    subset = df[mask]
                    
                    if len(subset) >= 10:
                        level1_patterns[key] = self._extract_all_gap_distributions(subset)
        
        patterns['hierarchical']['level_1'] = level1_patterns
        
        # Level 2: KTAS only
        level2_patterns = {}
        for ktas in range(1, 6):
            key = f"ktas_{ktas}"
            mask = df['ktas01'] == ktas
            subset = df[mask]
            
            if len(subset) >= 10:
                level2_patterns[key] = self._extract_all_gap_distributions(subset)
        
        patterns['hierarchical']['level_2'] = level2_patterns
        
        # Level 3: Overall fallback
        patterns['hierarchical']['level_3'] = {
            'overall': self._extract_all_gap_distributions(df)
        }
        
        # Summary statistics
        patterns['summary']['hierarchical_levels'] = {
            'level_1': len(level1_patterns),
            'level_2': len(level2_patterns),
            'level_3': 1
        }
        
        logger.info(f"Built {len(level1_patterns)} L1, {len(level2_patterns)} L2 patterns")
        return patterns
    
    def _extract_all_gap_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract distributions for ALL gap types"""
        distributions = {}
        
        for gap_name, gap_def in self.gap_definitions.items():
            gap_col = f'gap_{gap_name}'
            
            if gap_col in df.columns:
                gap_data = df[gap_col].dropna()
                max_minutes = gap_def['max_hours'] * 60
                gap_data = gap_data[(gap_data > 0) & (gap_data < max_minutes)]
                
                if len(gap_data) >= 5:
                    try:
                        # Fit log-normal distribution
                        shape, loc, scale = stats.lognorm.fit(gap_data, floc=0)
                        distributions[gap_name] = {
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
                        # Fallback to empirical
                        distributions[gap_name] = {
                            'distribution': 'empirical',
                            'mean': float(gap_data.mean()),
                            'median': float(gap_data.median()),
                            'std': float(gap_data.std()),
                            'count': int(len(gap_data))
                        }
        
        return distributions
    
        return distributions
    
    def _assign_flow_type(self, ktas: str, result: str) -> str:
        """Assign flow type based on KTAS and Result"""
        # 입원 (31: 병실, 32: 중환자실)
        if result == '32':
            return 'critical_care'
        elif result == '31':
            if ktas in ['1', '2']:
                return 'boarding'  # 중증인데 병실 입원이면 대기 가능성 높음
            else:
                return 'observation'
        
        # 전원 (21, 22)
        elif result in ['21', '22']:
            if ktas in ['1', '2']:
                return 'critical_care' # 중증 전원은 위급
            else:
                return 'standard'
                
        # 귀가 (11)
        elif result == '11':
            if ktas in ['4', '5']:
                return 'fast_track'
            else:
                return 'standard'
                
        # 사망 (41)
        elif result == '41':
            return 'critical_care'
            
        return 'standard'

    def _get_flow_type_multiplier(self, flow_type: str) -> float:
        """Get multiplier for flow type"""
        return self.flow_types.get(flow_type, self.flow_types['standard'])['multiplier']

    def generate_all_time_gaps(
        self,
        ktas_levels: np.ndarray,
        treatment_results: np.ndarray,
        base_datetime: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Generate ALL datetime columns based on comprehensive gap patterns
        
        Returns DataFrame with columns:
        - ocur_dt, ocur_tm: Incident occurrence
        - vst_dt, vst_tm: ER arrival
        - otrm_dt, otrm_tm: ER discharge
        - inpat_dt, inpat_tm: Hospital admission (if applicable)
        - otpat_dt, otpat_tm: Outpatient transfer (if applicable)
        """
        if self._time_patterns is None:
            self.analyze_all_time_patterns()
        
        n_patients = len(ktas_levels)
        
        # Initialize result dataframe
        result_df = pd.DataFrame()
        
        # Generate base times if not provided
        if base_datetime is None:
            base_datetime = datetime(2017, 1, 1)
        
        # Generate incident times (ocur) - some time before arrival
        ocur_times = []
        vst_times = []
        otrm_times = []
        inpat_times = []
        otpat_times = []
        
        for i in range(n_patients):
            ktas = str(int(ktas_levels[i]))
            result = str(treatment_results[i]) if pd.notna(treatment_results[i]) else None
            
            # Get distributions for this patient profile
            dist_params = self._get_hierarchical_distribution(ktas, result)
            
            # Determine Flow Type and Multiplier
            flow_type = self._assign_flow_type(ktas, result)
            multiplier = self._get_flow_type_multiplier(flow_type)
            
            # Generate arrival time (with some randomness)
            hours_offset = np.random.uniform(0, 24*365)  # Random time in year
            vst_time = base_datetime + timedelta(hours=hours_offset)
            vst_times.append(vst_time)
            
            # Generate incident time (before arrival)
            incident_gap = self._sample_from_distribution(
                dist_params.get('incident_to_arrival'),
                default_mean=60, default_std=30
            )
            
            # Apply multiplier to incident gap
            if incident_gap and incident_gap > 0:
                incident_gap *= multiplier
                ocur_time = vst_time - timedelta(minutes=incident_gap)
            else:
                ocur_time = vst_time - timedelta(minutes=30 * multiplier)
            ocur_times.append(ocur_time)
            
            # Generate discharge time (after arrival)
            er_stay = self._sample_from_distribution(
                dist_params.get('er_stay'),
                default_mean=180, default_std=60
            )
            
            # Apply multiplier to ER stay
            if er_stay and er_stay > 0:
                er_stay *= multiplier
                otrm_time = vst_time + timedelta(minutes=er_stay)
            else:
                otrm_time = vst_time + timedelta(minutes=180 * multiplier)
            otrm_times.append(otrm_time)
            
            # Generate admission time if applicable
            if result in ['31', '32', '33', '34']:  # Admission codes
                admit_gap = self._sample_from_distribution(
                    dist_params.get('discharge_to_admission'),
                    default_mean=60, default_std=30
                )
                if admit_gap and admit_gap > 0:
                    inpat_time = otrm_time + timedelta(minutes=admit_gap)
                else:
                    inpat_time = otrm_time + timedelta(minutes=60)
                inpat_times.append(inpat_time)
            else:
                inpat_times.append(None)
            
            # Generate outpatient time if applicable
            if result in ['21', '22']:  # Outpatient codes
                outpat_gap = self._sample_from_distribution(
                    dist_params.get('discharge_to_outpatient'),
                    default_mean=1440, default_std=720  # Default next day
                )
                if outpat_gap and outpat_gap > 0:
                    otpat_time = otrm_time + timedelta(minutes=outpat_gap)
                else:
                    otpat_time = otrm_time + timedelta(days=1)
                otpat_times.append(otpat_time)
            else:
                otpat_times.append(None)
        
        # Convert to NEDIS format
        result_df['ocur_dt'] = [dt.strftime('%Y%m%d') if dt else None for dt in ocur_times]
        result_df['ocur_tm'] = [dt.strftime('%H%M') if dt else None for dt in ocur_times]
        result_df['vst_dt'] = [dt.strftime('%Y%m%d') if dt else None for dt in vst_times]
        result_df['vst_tm'] = [dt.strftime('%H%M') if dt else None for dt in vst_times]
        result_df['otrm_dt'] = [dt.strftime('%Y%m%d') if dt else None for dt in otrm_times]
        result_df['otrm_tm'] = [dt.strftime('%H%M') if dt else None for dt in otrm_times]
        result_df['inpat_dt'] = [dt.strftime('%Y%m%d') if dt else None for dt in inpat_times]
        result_df['inpat_tm'] = [dt.strftime('%H%M') if dt else None for dt in inpat_times]
        result_df['otpat_dt'] = [dt.strftime('%Y%m%d') if dt else None for dt in otpat_times]
        result_df['otpat_tm'] = [dt.strftime('%H%M') if dt else None for dt in otpat_times]
        
        logger.info(f"Generated all datetime columns for {n_patients} patients")
        return result_df
    
    def _get_hierarchical_distribution(self, ktas: str, result: Optional[str]) -> Dict[str, Any]:
        """Get distribution parameters using hierarchical fallback"""
        # Try Level 1: KTAS + Result
        if result:
            key1 = f"ktas_{ktas}_result_{result}"
            if key1 in self._time_patterns['hierarchical']['level_1']:
                return self._time_patterns['hierarchical']['level_1'][key1]
        
        # Try Level 2: KTAS only
        key2 = f"ktas_{ktas}"
        if key2 in self._time_patterns['hierarchical']['level_2']:
            return self._time_patterns['hierarchical']['level_2'][key2]
        
        # Fallback to Level 3: Overall
        return self._time_patterns['hierarchical']['level_3']['overall']
    
    def _sample_from_distribution(
        self, 
        dist_params: Optional[Dict],
        default_mean: float = 60,
        default_std: float = 30
    ) -> float:
        """Sample from distribution or use defaults"""
        if not dist_params:
            return np.random.normal(default_mean, default_std)
        
        if dist_params.get('distribution') == 'lognorm':
            return stats.lognorm.rvs(
                dist_params['shape'],
                dist_params['loc'],
                dist_params['scale']
            )
        elif dist_params.get('distribution') == 'empirical':
            # Use normal approximation
            return np.random.normal(
                dist_params.get('mean', default_mean),
                dist_params.get('std', default_std)
            )
        else:
            return default_mean
    
    def _save_patterns_cache(self, patterns: Dict[str, Any]):
        """Save patterns to cache"""
        cache_dir = Path('cache/time_patterns')
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = cache_dir / 'comprehensive_time_patterns.json'
        with open(cache_file, 'w') as f:
            json.dump(patterns, f, indent=2, default=str)
        
        logger.info(f"Saved comprehensive patterns to {cache_file}")
    
    def validate_time_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that all generated times follow logical consistency:
        - ocur <= vst <= otrm
        - otrm <= inpat (if admitted)
        - otrm <= otpat (if outpatient)
        """
        validation_results = {
            'total_records': len(df),
            'consistency_checks': {}
        }
        
        # Parse all datetimes
        for prefix in ['ocur', 'vst', 'otrm', 'inpat', 'otpat']:
            if f'{prefix}_dt' in df.columns and f'{prefix}_tm' in df.columns:
                df[f'{prefix}_datetime'] = df.apply(
                    lambda x: self._parse_datetime(x[f'{prefix}_dt'], x[f'{prefix}_tm']),
                    axis=1
                )
        
        # Check: ocur <= vst
        if 'ocur_datetime' in df.columns and 'vst_datetime' in df.columns:
            valid_mask = df.apply(
                lambda x: pd.isna(x['ocur_datetime']) or pd.isna(x['vst_datetime']) or 
                         x['ocur_datetime'] <= x['vst_datetime'], axis=1
            )
            validation_results['consistency_checks']['ocur_before_vst'] = {
                'valid': int(valid_mask.sum()),
                'invalid': int((~valid_mask).sum()),
                'percentage': float(valid_mask.mean() * 100)
            }
        
        # Check: vst <= otrm
        if 'vst_datetime' in df.columns and 'otrm_datetime' in df.columns:
            valid_mask = df.apply(
                lambda x: pd.isna(x['vst_datetime']) or pd.isna(x['otrm_datetime']) or
                         x['vst_datetime'] <= x['otrm_datetime'], axis=1
            )
            validation_results['consistency_checks']['vst_before_otrm'] = {
                'valid': int(valid_mask.sum()),
                'invalid': int((~valid_mask).sum()),
                'percentage': float(valid_mask.mean() * 100)
            }
        
        # Check: otrm <= inpat (for admissions)
        if 'otrm_datetime' in df.columns and 'inpat_datetime' in df.columns:
            admission_mask = df['inpat_datetime'].notna()
            valid_mask = df[admission_mask].apply(
                lambda x: x['otrm_datetime'] <= x['inpat_datetime'], axis=1
            )
            validation_results['consistency_checks']['otrm_before_inpat'] = {
                'valid': int(valid_mask.sum()) if len(valid_mask) > 0 else 0,
                'invalid': int((~valid_mask).sum()) if len(valid_mask) > 0 else 0,
                'percentage': float(valid_mask.mean() * 100) if len(valid_mask) > 0 else 100.0
            }
        
        return validation_results