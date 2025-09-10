"""
Differential Privacy Implementation

Provides differential privacy mechanisms for synthetic data generation.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, List, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class NoiseType(Enum):
    """Types of noise mechanisms"""
    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"


class DifferentialPrivacy:
    """
    Implements differential privacy mechanisms
    """
    
    def __init__(self, epsilon: float = 1.0, delta: Optional[float] = None):
        """
        Initialize differential privacy mechanism
        
        Args:
            epsilon: Privacy budget (smaller = more privacy)
            delta: Delta parameter for (ε,δ)-differential privacy (for Gaussian noise)
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        
        self.epsilon = epsilon
        self.delta = delta
        self.consumed_budget = 0.0
        
        logger.info(f"Initialized DP with ε={epsilon}, δ={delta}")
    
    def add_laplace_noise(self, value: Union[float, np.ndarray], 
                         sensitivity: float) -> Union[float, np.ndarray]:
        """
        Add Laplace noise for differential privacy
        
        Args:
            value: Original value or array
            sensitivity: Sensitivity of the query
            
        Returns:
            Value with added Laplace noise
        """
        scale = sensitivity / self.epsilon
        
        if isinstance(value, np.ndarray):
            noise = np.random.laplace(0, scale, value.shape)
        else:
            noise = np.random.laplace(0, scale)
        
        noisy_value = value + noise
        
        # Track budget consumption
        self.consumed_budget += self.epsilon
        
        return noisy_value
    
    def add_gaussian_noise(self, value: Union[float, np.ndarray],
                          sensitivity: float,
                          delta: Optional[float] = None) -> Union[float, np.ndarray]:
        """
        Add Gaussian noise for (ε,δ)-differential privacy
        
        Args:
            value: Original value or array
            sensitivity: L2 sensitivity of the query
            delta: Delta parameter (uses instance delta if not provided)
            
        Returns:
            Value with added Gaussian noise
        """
        delta = delta or self.delta
        if delta is None or delta <= 0:
            raise ValueError("Delta must be positive for Gaussian noise")
        
        # Calculate standard deviation for Gaussian noise
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / self.epsilon
        
        if isinstance(value, np.ndarray):
            noise = np.random.normal(0, sigma, value.shape)
        else:
            noise = np.random.normal(0, sigma)
        
        noisy_value = value + noise
        
        # Track budget consumption
        self.consumed_budget += self.epsilon
        
        return noisy_value
    
    def exponential_mechanism(self, candidates: List[any],
                             scores: List[float],
                             sensitivity: float) -> any:
        """
        Exponential mechanism for selecting from discrete choices
        
        Args:
            candidates: List of candidate values
            scores: Utility scores for each candidate
            sensitivity: Sensitivity of the scoring function
            
        Returns:
            Selected candidate with differential privacy
        """
        if len(candidates) != len(scores):
            raise ValueError("Candidates and scores must have same length")
        
        # Calculate probabilities
        scores = np.array(scores)
        probabilities = np.exp(self.epsilon * scores / (2 * sensitivity))
        probabilities = probabilities / probabilities.sum()
        
        # Select candidate
        selected_idx = np.random.choice(len(candidates), p=probabilities)
        
        # Track budget consumption
        self.consumed_budget += self.epsilon
        
        return candidates[selected_idx]
    
    def private_count(self, true_count: int, 
                     sensitivity: int = 1) -> int:
        """
        Add noise to count query
        
        Args:
            true_count: True count value
            sensitivity: Sensitivity (usually 1 for single record contribution)
            
        Returns:
            Noisy count
        """
        noisy_count = self.add_laplace_noise(true_count, sensitivity)
        # Ensure non-negative count
        return max(0, int(np.round(noisy_count)))
    
    def private_sum(self, values: np.ndarray,
                   lower_bound: float,
                   upper_bound: float) -> float:
        """
        Add noise to sum query
        
        Args:
            values: Array of values to sum
            lower_bound: Lower bound of each value
            upper_bound: Upper bound of each value
            
        Returns:
            Noisy sum
        """
        true_sum = np.sum(values)
        sensitivity = upper_bound - lower_bound
        return self.add_laplace_noise(true_sum, sensitivity)
    
    def private_mean(self, values: np.ndarray,
                    lower_bound: float,
                    upper_bound: float) -> float:
        """
        Calculate differentially private mean
        
        Args:
            values: Array of values
            lower_bound: Lower bound of each value
            upper_bound: Upper bound of each value
            
        Returns:
            Noisy mean
        """
        # Use composition: noisy sum / noisy count
        # Split budget between sum and count
        original_epsilon = self.epsilon
        self.epsilon = original_epsilon / 2
        
        noisy_sum = self.private_sum(values, lower_bound, upper_bound)
        noisy_count = self.private_count(len(values))
        
        # Restore original epsilon
        self.epsilon = original_epsilon
        
        if noisy_count > 0:
            return noisy_sum / noisy_count
        else:
            return (upper_bound + lower_bound) / 2
    
    def private_histogram(self, data: pd.Series,
                         bins: Union[int, List] = 10,
                         density: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create differentially private histogram
        
        Args:
            data: Data series
            bins: Number of bins or bin edges
            density: If True, return density instead of counts
            
        Returns:
            Tuple of (counts/density, bin_edges)
        """
        # Create histogram
        counts, bin_edges = np.histogram(data, bins=bins)
        
        # Add noise to counts
        noisy_counts = self.add_laplace_noise(counts, sensitivity=1)
        
        # Ensure non-negative counts
        noisy_counts = np.maximum(0, noisy_counts)
        
        if density:
            # Convert to density
            width = bin_edges[1:] - bin_edges[:-1]
            total = np.sum(noisy_counts * width)
            if total > 0:
                noisy_counts = noisy_counts / total
        
        return noisy_counts, bin_edges
    
    def private_percentile(self, data: np.ndarray,
                          percentile: float,
                          lower_bound: float,
                          upper_bound: float) -> float:
        """
        Calculate differentially private percentile
        
        Args:
            data: Data array
            percentile: Percentile to calculate (0-100)
            lower_bound: Lower bound of data
            upper_bound: Upper bound of data
            
        Returns:
            Noisy percentile value
        """
        true_percentile = np.percentile(data, percentile)
        sensitivity = upper_bound - lower_bound
        
        # Add noise proportional to data range
        noisy_percentile = self.add_laplace_noise(true_percentile, sensitivity)
        
        # Clip to valid range
        return np.clip(noisy_percentile, lower_bound, upper_bound)
    
    def apply_to_dataframe(self, df: pd.DataFrame,
                          column_configs: Dict[str, Dict],
                          noise_type: NoiseType = NoiseType.LAPLACE) -> pd.DataFrame:
        """
        Apply differential privacy to dataframe columns
        
        Args:
            df: Input dataframe
            column_configs: Configuration for each column
                          e.g., {'age': {'sensitivity': 1, 'lower': 0, 'upper': 120}}
            noise_type: Type of noise to apply
            
        Returns:
            DataFrame with differential privacy applied
        """
        result_df = df.copy()
        
        for column, config in column_configs.items():
            if column not in df.columns:
                logger.warning(f"Column {column} not found in dataframe")
                continue
            
            sensitivity = config.get('sensitivity', 1)
            lower_bound = config.get('lower', df[column].min())
            upper_bound = config.get('upper', df[column].max())
            
            # Apply noise based on column type
            if pd.api.types.is_numeric_dtype(df[column]):
                if noise_type == NoiseType.LAPLACE:
                    result_df[column] = self.add_laplace_noise(
                        df[column].values, sensitivity
                    )
                elif noise_type == NoiseType.GAUSSIAN:
                    result_df[column] = self.add_gaussian_noise(
                        df[column].values, sensitivity
                    )
                
                # Clip to valid range
                result_df[column] = np.clip(
                    result_df[column], lower_bound, upper_bound
                )
                
                # Round if integer type
                if pd.api.types.is_integer_dtype(df[column]):
                    result_df[column] = np.round(result_df[column]).astype(int)
            
            else:
                logger.info(f"Skipping non-numeric column {column}")
        
        return result_df
    
    def get_privacy_budget_status(self) -> Dict[str, float]:
        """
        Get current privacy budget status
        
        Returns:
            Dictionary with budget information
        """
        return {
            'total_budget': self.epsilon,
            'consumed_budget': self.consumed_budget,
            'remaining_budget': max(0, self.epsilon - self.consumed_budget),
            'budget_exhausted': self.consumed_budget >= self.epsilon
        }
    
    def reset_budget(self):
        """Reset consumed privacy budget"""
        self.consumed_budget = 0.0
        logger.info("Privacy budget reset")


class PrivacyAccountant:
    """
    Tracks and manages privacy budget across multiple operations
    """
    
    def __init__(self, total_budget: float):
        """
        Initialize privacy accountant
        
        Args:
            total_budget: Total privacy budget available
        """
        self.total_budget = total_budget
        self.consumed_budget = 0.0
        self.operations = []
    
    def consume(self, epsilon: float, operation: str = "unknown") -> bool:
        """
        Consume privacy budget
        
        Args:
            epsilon: Amount of budget to consume
            operation: Description of operation
            
        Returns:
            True if budget was available, False otherwise
        """
        if self.consumed_budget + epsilon > self.total_budget:
            logger.warning(f"Insufficient privacy budget for {operation}")
            return False
        
        self.consumed_budget += epsilon
        self.operations.append({
            'operation': operation,
            'epsilon': epsilon,
            'cumulative': self.consumed_budget
        })
        
        logger.info(f"Consumed ε={epsilon} for {operation}, "
                   f"total consumed: {self.consumed_budget}/{self.total_budget}")
        
        return True
    
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget"""
        return max(0, self.total_budget - self.consumed_budget)
    
    def get_operations_log(self) -> List[Dict]:
        """Get log of all operations"""
        return self.operations.copy()
    
    def reset(self):
        """Reset the accountant"""
        self.consumed_budget = 0.0
        self.operations = []