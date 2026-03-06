"""
Comprehensive Time Gap Synthesizer for ALL NEDIS datetime pairs
Handles all non-overlapping time gaps between medical events
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
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
    - ptmiakdt/tm: Incident occurrence time
    - ptmiindt/tm: ER visit/arrival time  
    - ptmiotdt/tm: ER discharge time
    - ptmihsdt/tm: Hospital admission time
    - otpat_dt/tm: Outpatient transfer time
    """
    
    def __init__(self, db_manager: DatabaseManager, config: ConfigManager):
        self.db = db_manager
        self.config = config
        self.time_config = config.get('temporal', {})
        self._time_patterns = None
        
        # Prefix → NEDIS 4.0 column mapping (date, time)
        self._dt_col_map = {
            'vst':   ('ptmiindt', 'ptmiintm'),
            'ocur':  ('ptmiakdt', 'ptmiaktm'),
            'otrm':  ('ptmiotdt', 'ptmiottm'),
            'inpat': ('ptmihsdt', 'ptmihstm'),
        }

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
    
    def analyze_all_time_patterns(self) -> Dict[str, Any]:
        """
        Analyze ALL time gap patterns from real NEDIS data
        """
        logger.info("Analyzing comprehensive time gap patterns...")
        
        # Load data with all datetime columns
        source_table = self.config.get('original.source_table', 'nedis_original.emihptmi')
        
        # Build SELECT columns from the dt_col_map
        select_cols = ['ptmikpr1', 'ptmiemrt']
        for dt_col, tm_col in self._dt_col_map.values():
            select_cols.extend([dt_col, tm_col])

        query = f"""
        SELECT {', '.join(select_cols)}
        FROM {source_table}
        WHERE ptmikpr1 IS NOT NULL
            AND ptmikpr1 >= 1
            AND ptmikpr1 <= 5
        LIMIT 100000
        """

        df = self.db.fetch_dataframe(query)
        logger.info(f"Loaded {len(df)} records for analysis")

        # Parse all datetime pairs using NEDIS 4.0 column names
        for prefix, (dt_col, tm_col) in self._dt_col_map.items():
            df[f'{prefix}_datetime'] = df.apply(
                lambda x, d=dt_col, t=tm_col: self._parse_datetime(x[d], x[t]),
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
            for result in df['ptmiemrt'].unique():
                if pd.notna(result):
                    key = f"ktas_{ktas}_result_{result}"
                    mask = (df['ptmikpr1'] == ktas) & (df['ptmiemrt'] == result)
                    subset = df[mask]
                    
                    if len(subset) >= 10:
                        level1_patterns[key] = self._extract_all_gap_distributions(subset)
        
        patterns['hierarchical']['level_1'] = level1_patterns
        
        # Level 2: KTAS only
        level2_patterns = {}
        for ktas in range(1, 6):
            key = f"ktas_{ktas}"
            mask = df['ptmikpr1'] == ktas
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
    
    def generate_all_time_gaps(
        self,
        ktas_levels: np.ndarray,
        treatment_results: np.ndarray,
        base_datetime: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Generate ALL datetime columns based on comprehensive gap patterns
        
        Returns DataFrame with columns:
        - ptmiakdt, ptmiaktm: Incident occurrence
        - ptmiindt, ptmiintm: ER arrival
        - ptmiotdt, ptmiottm: ER discharge
        - ptmihsdt, ptmihstm: Hospital admission (if applicable)
        - otpat_dt, otpat_tm: Outpatient transfer (if applicable)
        """
        if self._time_patterns is None:
            self.analyze_all_time_patterns()

        n_patients = len(ktas_levels)
        base_arrivals = self._fill_missing_arrivals(
            self._coerce_base_arrivals(base_datetime, n_patients)
        )
        
        # Initialize result dataframe
        result_df = pd.DataFrame()

        # Generate incident times (ocur) - some time before arrival
        ocur_times = []
        vst_times = []
        otrm_times = []
        inpat_times = []
        otpat_times = []
        
        for i in range(n_patients):
            ktas = self._normalize_code(ktas_levels[i], allow_none=True)
            result = self._normalize_code(treatment_results[i], allow_none=True)
            
            # Get distributions for this patient profile
            dist_params = self._get_hierarchical_distribution(ktas, result)
            
            vst_time = base_arrivals[i]
            vst_times.append(vst_time)
            
            # Generate incident time (before arrival)
            incident_gap = self._sample_gap_minutes(
                dist_params.get('incident_to_arrival'),
                gap_name='incident_to_arrival',
                default_mean=45,
                default_std=20,
                ktas=ktas
            )
            ocur_time = vst_time - timedelta(minutes=incident_gap)
            ocur_times.append(ocur_time)
            
            # Generate discharge time (after arrival)
            er_stay = self._sample_gap_minutes(
                dist_params.get('er_stay'),
                gap_name='er_stay',
                default_mean=180,
                default_std=60,
                ktas=ktas
            )
            otrm_time = vst_time + timedelta(minutes=er_stay)
            otrm_times.append(otrm_time)
            
            # Generate admission time if applicable
            if result in ['31', '32', '33', '34']:  # Admission codes
                admit_gap = self._sample_gap_minutes(
                    dist_params.get('discharge_to_admission'),
                    gap_name='discharge_to_admission',
                    default_mean=60,
                    default_std=30,
                    ktas=ktas
                )
                inpat_time = otrm_time + timedelta(minutes=admit_gap)
                inpat_times.append(inpat_time)
            else:
                inpat_times.append(None)
            
            # Generate outpatient time if applicable
            if result in ['21', '22']:  # Outpatient codes
                outpat_gap = self._sample_gap_minutes(
                    dist_params.get('discharge_to_outpatient'),
                    gap_name='discharge_to_outpatient',
                    default_mean=1440,
                    default_std=720,
                    ktas=ktas  # Default next day
                )
                otpat_time = otrm_time + timedelta(minutes=outpat_gap)
                otpat_times.append(otpat_time)
            else:
                otpat_times.append(None)
        
        # Convert to NEDIS format
        result_df['ptmiakdt'] = [dt.strftime('%Y%m%d') if dt else None for dt in ocur_times]
        result_df['ptmiaktm'] = [dt.strftime('%H%M') if dt else None for dt in ocur_times]
        result_df['ptmiindt'] = [dt.strftime('%Y%m%d') if dt else None for dt in vst_times]
        result_df['ptmiintm'] = [dt.strftime('%H%M') if dt else None for dt in vst_times]
        result_df['ptmiotdt'] = [dt.strftime('%Y%m%d') if dt else None for dt in otrm_times]
        result_df['ptmiottm'] = [dt.strftime('%H%M') if dt else None for dt in otrm_times]
        result_df['ptmihsdt'] = [dt.strftime('%Y%m%d') if dt else None for dt in inpat_times]
        result_df['ptmihstm'] = [dt.strftime('%H%M') if dt else None for dt in inpat_times]
        result_df['otpat_dt'] = [dt.strftime('%Y%m%d') if dt else None for dt in otpat_times]
        result_df['otpat_tm'] = [dt.strftime('%H%M') if dt else None for dt in otpat_times]
        
        logger.info(f"Generated all datetime columns for {n_patients} patients")
        return result_df

    def _normalize_code(self, value: Any, allow_none: bool = False) -> Optional[str]:
        """일관된 문자열 코드로 정규화"""
        if pd.isna(value):
            return None if allow_none else "0"

        code = str(value).strip()
        if not code:
            return None if allow_none else "0"

        if code.lower() in {"nan", "none", "null"}:
            return None if allow_none else "0"

        if "." in code:
            code = code.split(".")[0]
        if code.endswith(".0"):
            code = code[:-2]

        return code

    def _coerce_base_arrivals(
        self,
        base_datetime: Optional[Any],
        n_patients: int
    ) -> List[Optional[datetime]]:
        """입력된 기본 도착 시각을 NEDIS 생성용 datetime 리스트로 정규화"""
        if base_datetime is None:
            return [None] * n_patients

        if isinstance(base_datetime, (datetime, pd.Timestamp)):
            return [pd.Timestamp(base_datetime).to_pydatetime()] * n_patients

        try:
            base_series = pd.to_datetime(base_datetime, errors='coerce')
            if isinstance(base_series, pd.Series):
                normalized = base_series.reset_index(drop=True)
            else:
                normalized = pd.Series(base_series).reset_index(drop=True)
        except Exception:
            normalized = pd.Series([pd.NaT] * n_patients)

        if len(normalized) < n_patients:
            normalized = normalized.reindex(range(n_patients))
        elif len(normalized) > n_patients:
            normalized = normalized.iloc[:n_patients]

        return [v.to_pydatetime() if pd.notna(v) else None for v in normalized]

    def _fill_missing_arrivals(self, base_arrivals: List[Optional[datetime]]) -> List[datetime]:
        """결측 도착 시각을 학습된 분포에서 보완"""
        if not base_arrivals:
            return []

        valid = [v for v in base_arrivals if v is not None]

        # 결측이 없으면 그대로 반환
        if not valid:
            fallback_start = datetime(2017, 1, 1)
            max_minutes = 365 * 24 * 60
            return [fallback_start + timedelta(minutes=np.random.uniform(0, max_minutes)) for _ in base_arrivals]

        filled = []
        for arrival in base_arrivals:
            if arrival is None:
                filled.append(np.random.choice(valid))
            else:
                filled.append(arrival)
        return filled
    
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
    
    def _sample_gap_minutes(
        self, 
        dist_params: Optional[Dict],
        gap_name: str,
        default_mean: float = 60,
        default_std: float = 30,
        ktas: str = "3"
    ) -> float:
        """샘플링한 시간 차이(분)를 반환하고 비정상 값 보정"""
        max_minutes = self.gap_definitions.get(gap_name, {}).get('max_hours', 24) * 60
        if max_minutes <= 0:
            max_minutes = 24 * 60

        defaults = self._get_gap_defaults(gap_name, ktas, default_mean, default_std)
        if not dist_params:
            sampled = np.random.normal(defaults["mean"], defaults["std"])
            return float(np.clip(sampled, 1, max_minutes))
        
        if dist_params.get('distribution') == 'lognorm':
            sampled = stats.lognorm.rvs(
                dist_params['shape'],
                dist_params['loc'],
                dist_params['scale']
            )
        elif dist_params.get('distribution') == 'empirical':
            # Use normal approximation
            sampled = np.random.normal(
                dist_params.get('mean', defaults["mean"]),
                dist_params.get('std', defaults["std"])
            )
        else:
            sampled = defaults["mean"]

        if not np.isfinite(sampled):
            sampled = defaults["mean"]

        return float(np.clip(float(sampled), 1, max_minutes))

    def _get_gap_defaults(self, gap_name: str, ktas: str, default_mean: float, default_std: float) -> Dict[str, float]:
        """KTAS별 gap 기본값"""
        ktas_key = (ktas or "3")[0]

        severity_defaults = {
            "incident_to_arrival": {
                "mean_by_level": {"1": 30, "2": 45, "3": 60, "4": 90, "5": 120},
                "std_by_level": {"1": 15, "2": 20, "3": 25, "4": 35, "5": 40},
            },
            "er_stay": {
                "mean_by_level": {"1": 240, "2": 210, "3": 180, "4": 150, "5": 120},
                "std_by_level": {"1": 90, "2": 80, "3": 70, "4": 60, "5": 80},
            },
            "discharge_to_admission": {
                "mean_by_level": {"1": 30, "2": 45, "3": 60, "4": 90, "5": 120},
                "std_by_level": {"1": 20, "2": 30, "3": 40, "4": 60, "5": 90},
            },
            "discharge_to_outpatient": {
                "mean_by_level": {"1": 120, "2": 360, "3": 720, "4": 1080, "5": 1440},
                "std_by_level": {"1": 60, "2": 120, "3": 240, "4": 360, "5": 480},
            },
            "arrival_to_admission": {
                "mean_by_level": {"1": 45, "2": 90, "3": 150, "4": 210, "5": 300},
                "std_by_level": {"1": 30, "2": 45, "3": 60, "4": 90, "5": 120},
            },
            "incident_to_discharge": {
                "mean_by_level": {"1": 270, "2": 255, "3": 240, "4": 210, "5": 180},
                "std_by_level": {"1": 120, "2": 110, "3": 100, "4": 90, "5": 90},
            },
        }

        cfg = severity_defaults.get(gap_name, {})
        mean_map = cfg.get("mean_by_level", {})
        std_map = cfg.get("std_by_level", {})
        return {
            "mean": float(mean_map.get(ktas_key, default_mean)),
            "std": float(std_map.get(ktas_key, default_std)),
        }
    
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
