"""
Identifier Management for Privacy Protection

Handles generation of synthetic identifiers and removal of direct identifiers.
"""

import hashlib
import secrets
import time
from typing import List, Dict, Optional, Set
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class IdentifierManager:
    """
    Manages synthetic identifier generation and direct identifier removal
    """
    
    def __init__(self, salt: Optional[str] = None):
        """
        Initialize identifier manager
        
        Args:
            salt: Optional salt for consistent hashing (for reproducibility)
        """
        self.salt = salt or secrets.token_hex(32)
        self.generated_ids: Set[str] = set()
        
        # Direct identifiers to remove/replace
        self.direct_identifiers = [
            'pat_reg_no',     # Patient registration number
            'index_key',      # Original index
            'pat_brdt',       # Birth date (convert to age)
            'emorg_cd',       # Hospital code (may need generalization)
        ]
        
        # Quasi-identifiers that need generalization
        self.quasi_identifiers = [
            'pat_age',        # Age
            'pat_sex',        # Gender
            'pat_sarea',      # Area code
            'vst_dt',         # Visit date
            'vst_tm',         # Visit time
        ]
    
    def generate_unique_id(self, prefix: str = "SYN") -> str:
        """
        Generate a unique synthetic identifier
        
        Args:
            prefix: Prefix for the ID (default: "SYN" for synthetic)
            
        Returns:
            Unique identifier string
        """
        # Combine timestamp, random bytes, and counter for uniqueness
        timestamp = str(time.time_ns())
        random_bytes = secrets.token_bytes(16)
        
        # Create hash
        hash_input = f"{prefix}_{timestamp}_{random_bytes.hex()}_{self.salt}"
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()
        
        # Create readable ID: PREFIX_YYYYMMDD_HASH8
        date_str = datetime.now().strftime("%Y%m%d")
        synthetic_id = f"{prefix}_{date_str}_{hash_value[:8].upper()}"
        
        # Ensure uniqueness
        while synthetic_id in self.generated_ids:
            random_bytes = secrets.token_bytes(16)
            hash_input = f"{prefix}_{timestamp}_{random_bytes.hex()}_{self.salt}"
            hash_value = hashlib.sha256(hash_input.encode()).hexdigest()
            synthetic_id = f"{prefix}_{date_str}_{hash_value[:8].upper()}"
        
        self.generated_ids.add(synthetic_id)
        return synthetic_id
    
    def generate_batch_ids(self, count: int, prefix: str = "SYN") -> List[str]:
        """
        Generate multiple unique IDs efficiently
        
        Args:
            count: Number of IDs to generate
            prefix: Prefix for the IDs
            
        Returns:
            List of unique identifiers
        """
        ids = []
        for i in range(count):
            ids.append(self.generate_unique_id(prefix))
        return ids
    
    def anonymize_dataframe(self, df: pd.DataFrame, 
                           id_column: str = 'synthetic_id',
                           preserve_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Anonymize a dataframe by replacing/removing direct identifiers
        
        Args:
            df: Input dataframe
            id_column: Name for the new synthetic ID column
            preserve_columns: Columns to preserve (not remove)
            
        Returns:
            Anonymized dataframe
        """
        df = df.copy()
        preserve_columns = preserve_columns or []
        
        # Generate new synthetic IDs
        logger.info(f"Generating {len(df)} synthetic identifiers")
        df[id_column] = self.generate_batch_ids(len(df))
        
        # Remove direct identifiers
        columns_to_remove = []
        for col in self.direct_identifiers:
            if col in df.columns and col not in preserve_columns:
                columns_to_remove.append(col)
        
        if columns_to_remove:
            logger.info(f"Removing direct identifiers: {columns_to_remove}")
            df = df.drop(columns=columns_to_remove)
        
        # Convert birth date to age if present
        if 'pat_brdt' in df.columns and 'pat_brdt' not in preserve_columns:
            logger.info("Converting birth date to age")
            df = self._convert_birthdate_to_age(df)
        
        return df
    
    def _convert_birthdate_to_age(self, df: pd.DataFrame, 
                                  reference_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Convert birth date to age
        
        Args:
            df: Dataframe with pat_brdt column
            reference_date: Reference date for age calculation
            
        Returns:
            Dataframe with age instead of birth date
        """
        if 'pat_brdt' not in df.columns:
            return df
        
        reference_date = reference_date or datetime(2017, 12, 31)
        
        # Parse birth dates (handle various formats)
        def parse_birthdate(date_str):
            if pd.isna(date_str):
                return None
            
            date_str = str(date_str).strip()
            
            # Try different formats
            for fmt in ['%Y%m%d', '%y%m%d', '%Y-%m-%d', '%y-%m-%d']:
                try:
                    birth_date = datetime.strptime(date_str, fmt)
                    # Handle 2-digit years
                    if birth_date.year < 100:
                        birth_date = birth_date.replace(year=birth_date.year + 1900)
                    return birth_date
                except:
                    continue
            
            return None
        
        # Calculate ages
        birth_dates = df['pat_brdt'].apply(parse_birthdate)
        ages = birth_dates.apply(
            lambda bd: (reference_date - bd).days // 365 if bd else None
        )
        
        # Replace birth date with age if not already present
        if 'pat_age' not in df.columns:
            df['pat_age'] = ages
        
        # Remove birth date column
        df = df.drop(columns=['pat_brdt'])
        
        return df
    
    def create_mapping_table(self, original_ids: List[str], 
                           synthetic_ids: List[str]) -> pd.DataFrame:
        """
        Create a secure mapping table between original and synthetic IDs
        
        Args:
            original_ids: List of original identifiers
            synthetic_ids: List of synthetic identifiers
            
        Returns:
            Mapping dataframe (should be stored securely)
        """
        if len(original_ids) != len(synthetic_ids):
            raise ValueError("ID lists must have the same length")
        
        mapping_df = pd.DataFrame({
            'original_id': original_ids,
            'synthetic_id': synthetic_ids,
            'created_at': datetime.now(),
            'hash_salt': self.salt
        })
        
        # Add integrity check
        mapping_df['integrity_hash'] = mapping_df.apply(
            lambda row: hashlib.sha256(
                f"{row['original_id']}_{row['synthetic_id']}_{self.salt}".encode()
            ).hexdigest()[:16],
            axis=1
        )
        
        return mapping_df
    
    def validate_anonymization(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate that direct identifiers have been properly removed/anonymized
        
        Args:
            df: Dataframe to validate
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        # Check for presence of direct identifiers
        for identifier in self.direct_identifiers:
            results[f"removed_{identifier}"] = identifier not in df.columns
        
        # Check for synthetic ID column
        results['has_synthetic_id'] = any('synthetic' in col.lower() for col in df.columns)
        
        # Check age conversion
        results['has_age_not_birthdate'] = 'pat_age' in df.columns and 'pat_brdt' not in df.columns
        
        # Overall validation
        results['valid'] = all([
            results[f"removed_{id}"] for id in self.direct_identifiers
            if f"removed_{id}" in results
        ]) and results['has_synthetic_id']
        
        return results