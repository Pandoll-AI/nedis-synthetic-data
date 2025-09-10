"""
Generalization Functions for Quasi-Identifiers

Implements age and geographic generalization to reduce re-identification risk.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AgeGeneralizer:
    """
    Generalizes age values to reduce granularity
    """
    
    def __init__(self, group_size: int = 10, 
                 special_groups: Optional[Dict[Tuple[int, int], int]] = None):
        """
        Initialize age generalizer
        
        Args:
            group_size: Default age group size (e.g., 10 for decades)
            special_groups: Special age ranges with different group sizes
                           e.g., {(0, 2): 1, (90, 120): 30}
        """
        self.group_size = group_size
        self.special_groups = special_groups or {
            (0, 2): 1,      # Keep infant granularity
            (90, 120): 30   # Group all 90+ together
        }
    
    def generalize(self, age: int, method: str = 'random') -> int:
        """
        Generalize a single age value
        
        Args:
            age: Original age
            method: 'random' (random within group), 'center' (group center), 
                   'lower' (group start), 'upper' (group end)
            
        Returns:
            Generalized age
        """
        if pd.isna(age) or age < 0:
            return age
        
        # Check special groups
        for (min_age, max_age), special_size in self.special_groups.items():
            if min_age <= age <= max_age:
                if special_size == 1:
                    return age  # Keep as is
                else:
                    # Use special group size
                    group_start = min_age
                    group_end = max_age
                    break
        else:
            # Use default grouping
            group_start = (age // self.group_size) * self.group_size
            group_end = group_start + self.group_size - 1
        
        # Apply generalization method
        if method == 'random':
            return np.random.randint(group_start, min(group_end + 1, 120))
        elif method == 'center':
            return (group_start + group_end) // 2
        elif method == 'lower':
            return group_start
        elif method == 'upper':
            return min(group_end, 120)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def generalize_series(self, ages: pd.Series, 
                         method: str = 'random',
                         preserve_distribution: bool = True) -> pd.Series:
        """
        Generalize a series of ages
        
        Args:
            ages: Series of age values
            method: Generalization method
            preserve_distribution: Try to preserve age distribution within groups
            
        Returns:
            Series of generalized ages
        """
        if preserve_distribution and method == 'random':
            # Group ages and sample preserving distribution
            generalized = pd.Series(index=ages.index, dtype='float64')
            
            for age_group in ages.value_counts().index:
                if pd.isna(age_group):
                    continue
                    
                mask = ages == age_group
                count = mask.sum()
                
                # Determine group range
                group_start = (age_group // self.group_size) * self.group_size
                group_end = group_start + self.group_size - 1
                
                # Check for special groups
                for (min_age, max_age), special_size in self.special_groups.items():
                    if min_age <= age_group <= max_age:
                        if special_size == 1:
                            generalized[mask] = age_group
                        else:
                            generalized[mask] = np.random.randint(min_age, max_age + 1, count)
                        break
                else:
                    # Generate random ages within group
                    generalized[mask] = np.random.randint(group_start, 
                                                         min(group_end + 1, 120), 
                                                         count)
            
            return generalized
        else:
            # Simple application
            return ages.apply(lambda x: self.generalize(x, method))
    
    def get_age_groups(self, ages: pd.Series) -> pd.Series:
        """
        Get age group labels for a series of ages
        
        Args:
            ages: Series of age values
            
        Returns:
            Series of age group labels (e.g., "20-29")
        """
        def get_group_label(age):
            if pd.isna(age) or age < 0:
                return "Unknown"
            
            # Check special groups
            for (min_age, max_age), special_size in self.special_groups.items():
                if min_age <= age <= max_age:
                    if special_size == 1:
                        return str(age)
                    else:
                        return f"{min_age}-{max_age}"
            
            # Default grouping
            group_start = (age // self.group_size) * self.group_size
            group_end = group_start + self.group_size - 1
            return f"{group_start}-{group_end}"
        
        return ages.apply(get_group_label)


class GeographicGeneralizer:
    """
    Generalizes geographic identifiers (region codes)
    """
    
    def __init__(self, hierarchy_levels: Optional[Dict[str, int]] = None,
                 population_thresholds: Optional[Dict[int, int]] = None):
        """
        Initialize geographic generalizer
        
        Args:
            hierarchy_levels: Mapping of level names to digit counts
                            e.g., {'province': 2, 'district': 4, 'detail': 6}
            population_thresholds: Population thresholds for each level
                                  e.g., {2: 100000, 4: 50000, 6: 10000}
        """
        self.hierarchy_levels = hierarchy_levels or {
            'province': 2,
            'district': 4,
            'detail': 6
        }
        
        self.population_thresholds = population_thresholds or {
            2: 100000,  # Use province level if pop < 100k
            4: 50000,   # Use district level if pop < 50k
            6: 10000    # Use detail level if pop < 10k
        }
        
        # Region population data (would be loaded from actual data)
        self.region_populations = {}
    
    def load_population_data(self, population_df: pd.DataFrame):
        """
        Load population data for regions
        
        Args:
            population_df: DataFrame with columns 'region_code' and 'population'
        """
        self.region_populations = dict(zip(
            population_df['region_code'].astype(str),
            population_df['population']
        ))
    
    def generalize(self, region_code: str, 
                  target_level: Optional[str] = None,
                  use_population: bool = True) -> str:
        """
        Generalize a region code
        
        Args:
            region_code: Original region code
            target_level: Target generalization level ('province', 'district', 'detail')
            use_population: Use population-based suppression
            
        Returns:
            Generalized region code
        """
        if pd.isna(region_code):
            return region_code
        
        region_code = str(region_code).strip()
        
        # Determine target level based on population if requested
        if use_population and region_code in self.region_populations:
            population = self.region_populations[region_code]
            
            # Determine appropriate level based on population
            if population < self.population_thresholds[6]:
                target_level = 'province'  # Highest generalization
            elif population < self.population_thresholds[4]:
                target_level = 'district'
            else:
                target_level = 'detail'  # Keep original
        
        # Apply generalization
        if target_level:
            target_digits = self.hierarchy_levels.get(target_level, 6)
            return region_code[:target_digits]
        
        # Default: return district level (4 digits)
        return region_code[:4] if len(region_code) >= 4 else region_code
    
    def generalize_series(self, regions: pd.Series, 
                         target_level: Optional[str] = None,
                         use_population: bool = True) -> pd.Series:
        """
        Generalize a series of region codes
        
        Args:
            regions: Series of region codes
            target_level: Target generalization level
            use_population: Use population-based suppression
            
        Returns:
            Series of generalized region codes
        """
        return regions.apply(
            lambda x: self.generalize(x, target_level, use_population)
        )
    
    def get_hierarchy_level(self, region_code: str) -> str:
        """
        Determine the hierarchy level of a region code
        
        Args:
            region_code: Region code
            
        Returns:
            Hierarchy level name
        """
        if pd.isna(region_code):
            return "unknown"
        
        code_length = len(str(region_code).strip())
        
        for level_name, digits in sorted(self.hierarchy_levels.items(), 
                                        key=lambda x: x[1]):
            if code_length <= digits:
                return level_name
        
        return "detail"
    
    def suppress_rare_regions(self, regions: pd.Series, 
                             min_count: int = 5,
                             suppression_value: str = "OTHER") -> pd.Series:
        """
        Suppress rare regions that appear less than min_count times
        
        Args:
            regions: Series of region codes
            min_count: Minimum count threshold
            suppression_value: Value to use for suppressed regions
            
        Returns:
            Series with rare regions suppressed
        """
        region_counts = regions.value_counts()
        rare_regions = region_counts[region_counts < min_count].index
        
        if len(rare_regions) > 0:
            logger.info(f"Suppressing {len(rare_regions)} rare regions")
            regions = regions.copy()
            regions[regions.isin(rare_regions)] = suppression_value
        
        return regions


class TemporalGeneralizer:
    """
    Generalizes temporal identifiers (dates and times)
    """
    
    def __init__(self, time_units: Optional[Dict[str, int]] = None):
        """
        Initialize temporal generalizer
        
        Args:
            time_units: Mapping of unit names to minutes
                       e.g., {'hour': 60, 'shift': 480, 'day': 1440}
        """
        self.time_units = time_units or {
            'minute': 1,
            'quarter_hour': 15,
            'half_hour': 30,
            'hour': 60,
            'shift': 480,      # 8-hour shift
            'day': 1440,       # 24 hours
            'week': 10080      # 7 days
        }
    
    def round_time(self, time_str: str, unit: str = 'hour') -> str:
        """
        Round a time string to specified unit
        
        Args:
            time_str: Time string (HHMM or HHMMSS format)
            unit: Time unit to round to
            
        Returns:
            Rounded time string
        """
        if pd.isna(time_str):
            return time_str
        
        time_str = str(time_str).strip().zfill(4)
        
        try:
            hours = int(time_str[:2])
            minutes = int(time_str[2:4])
            total_minutes = hours * 60 + minutes
            
            # Round to specified unit
            unit_minutes = self.time_units.get(unit, 60)
            rounded_minutes = round(total_minutes / unit_minutes) * unit_minutes
            
            # Handle day overflow
            rounded_minutes = rounded_minutes % 1440
            
            # Convert back to HHMM format
            new_hours = rounded_minutes // 60
            new_minutes = rounded_minutes % 60
            
            return f"{new_hours:02d}{new_minutes:02d}"
        except:
            return time_str
    
    def add_noise(self, time_str: str, max_shift_minutes: int = 30) -> str:
        """
        Add random noise to time
        
        Args:
            time_str: Time string (HHMM format)
            max_shift_minutes: Maximum shift in minutes
            
        Returns:
            Time with added noise
        """
        if pd.isna(time_str):
            return time_str
        
        time_str = str(time_str).strip().zfill(4)
        
        try:
            hours = int(time_str[:2])
            minutes = int(time_str[2:4])
            total_minutes = hours * 60 + minutes
            
            # Add random noise
            shift = np.random.randint(-max_shift_minutes, max_shift_minutes + 1)
            new_minutes = total_minutes + shift
            
            # Handle day boundaries
            new_minutes = max(0, min(1439, new_minutes))
            
            # Convert back to HHMM format
            new_hours = new_minutes // 60
            new_mins = new_minutes % 60
            
            return f"{new_hours:02d}{new_mins:02d}"
        except:
            return time_str