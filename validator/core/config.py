"""
Configuration management for the validation system.

This module handles all configuration aspects including:
- YAML/JSON config file parsing
- Environment variable integration
- Configuration validation
- Default value management
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ValidationConfig:
    """Main configuration class for the validation system"""

    # 검증 설정
    significance_level: float = 0.05
    sample_size: int = 50000
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds

    # 통계적 검증 설정
    statistics: Dict[str, Any] = field(default_factory=lambda: {
        "ks_threshold": 0.05,
        "chi2_threshold": 0.05,
        "correlation_threshold": 0.1,
        "wasserstein_threshold": 0.1,
        "min_sample_size": 10
    })

    # 패턴 분석 설정
    patterns: Dict[str, Any] = field(default_factory=lambda: {
        "min_sample_size": 10,
        "confidence_threshold": 0.95,
        "hierarchical_fallback": True,
        "time_gap_analysis": True,
        "pattern_cache_enabled": True
    })

    # 시각화 설정
    visualization: Dict[str, Any] = field(default_factory=lambda: {
        "enable_dashboard": True,
        "dashboard_port": 8050,
        "report_formats": ["html", "pdf", "json"],
        "chart_theme": "default",
        "interactive_charts": True
    })

    # API 설정
    api: Dict[str, Any] = field(default_factory=lambda: {
        "host": "0.0.0.0",
        "port": 8000,
        "enable_cors": True,
        "rate_limit": 100,
        "enable_graphql": True,
        "enable_websocket": True
    })

    # 데이터베이스 설정
    databases: Dict[str, Any] = field(default_factory=lambda: {
        "original": {
            "path": "nedis_data.duckdb",
            "schema": "main"
        },
        "synthetic": {
            "path": "nedis_synth_2017.duckdb",
            "schema": "main"
        }
    })

    # 보안 설정
    security: Dict[str, Any] = field(default_factory=lambda: {
        "enable_privacy_protection": True,
        "differential_privacy_epsilon": 1.0,
        "enable_pii_masking": True,
        "enable_audit_logging": True
    })

    # 성능 설정
    performance: Dict[str, Any] = field(default_factory=lambda: {
        "max_concurrent_validations": 3,
        "timeout_seconds": 300,
        "memory_limit_mb": 2048,
        "enable_profiling": False
    })

    @classmethod
    def from_yaml(cls, config_path: str) -> 'ValidationConfig':
        """Load configuration from YAML file"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ValidationConfig':
        """Load configuration from dictionary"""
        # Create instance with default values
        config = cls()

        # Update with provided values
        for key, value in config_dict.items():
            if hasattr(config, key):
                if isinstance(value, dict) and hasattr(config, key):
                    # Merge nested dictionaries
                    current_value = getattr(config, key)
                    if isinstance(current_value, dict):
                        current_value.update(value)
                    else:
                        setattr(config, key, value)
                else:
                    setattr(config, key, value)

        return config

    @classmethod
    def from_env(cls) -> 'ValidationConfig':
        """Load configuration from environment variables"""
        config = cls()

        # Environment variable mappings
        env_mappings = {
            'significance_level': 'VALIDATION_SIGNIFICANCE_LEVEL',
            'sample_size': 'VALIDATION_SAMPLE_SIZE',
            'enable_caching': 'VALIDATION_ENABLE_CACHING',
            'cache_ttl': 'VALIDATION_CACHE_TTL',
            'dashboard_port': 'VALIDATION_DASHBOARD_PORT',
            'api_port': 'VALIDATION_API_PORT',
            'enable_privacy_protection': 'VALIDATION_ENABLE_PRIVACY'
        }

        for attr, env_var in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                # Convert string values to appropriate types
                if attr in ['significance_level', 'cache_ttl']:
                    value = float(value)
                elif attr in ['sample_size', 'dashboard_port', 'api_port']:
                    value = int(value)
                elif attr in ['enable_caching', 'enable_privacy_protection']:
                    value = value.lower() in ('true', '1', 'yes', 'on')

                setattr(config, attr, value)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'significance_level': self.significance_level,
            'sample_size': self.sample_size,
            'enable_caching': self.enable_caching,
            'cache_ttl': self.cache_ttl,
            'statistics': self.statistics,
            'patterns': self.patterns,
            'visualization': self.visualization,
            'api': self.api,
            'databases': self.databases,
            'security': self.security,
            'performance': self.performance
        }

    def save(self, config_path: str):
        """Save configuration to YAML file"""
        config_path = Path(config_path)

        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True, indent=2)

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []

        # Validate ranges
        if not 0 < self.significance_level <= 1:
            errors.append("significance_level must be between 0 and 1")

        if self.sample_size < 1:
            errors.append("sample_size must be positive")

        if self.cache_ttl < 0:
            errors.append("cache_ttl must be non-negative")

        # Validate nested configs
        for key, config in self.statistics.items():
            if isinstance(config, (int, float)) and config < 0:
                errors.append(f"statistics.{key} must be non-negative")

        for key, config in self.patterns.items():
            if key == 'confidence_threshold' and not 0 <= config <= 1:
                errors.append("patterns.confidence_threshold must be between 0 and 1")
            elif key == 'min_sample_size' and config < 1:
                errors.append("patterns.min_sample_size must be positive")

        # Validate database paths (skip validation for demo purposes)
        # for db_type in ['original', 'synthetic']:
        #     db_config = self.databases.get(db_type, {})
        #     if 'path' in db_config:
        #         db_path = Path(db_config['path'])
        #         if not db_path.exists():
        #             errors.append(f"Database path does not exist: {db_path}")

        return errors

    def get_database_config(self, db_type: str) -> Dict[str, Any]:
        """Get database configuration for specific type"""
        return self.databases.get(db_type, {})

    def get_statistical_config(self, key: str, default: Any = None) -> Any:
        """Get statistical configuration value"""
        return self.statistics.get(key, default)

    def get_pattern_config(self, key: str, default: Any = None) -> Any:
        """Get pattern analysis configuration value"""
        return self.patterns.get(key, default)


class ConfigManager:
    """Configuration manager with caching and validation"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._config: Optional[ValidationConfig] = None
        self._last_modified: Optional[datetime] = None

    def get_config(self) -> ValidationConfig:
        """Get configuration with caching"""
        if self._config is None:
            self._config = self._load_config()
        elif self._should_reload():
            self._config = self._load_config()

        return self._config

    def _load_config(self) -> ValidationConfig:
        """Load configuration from various sources"""
        config = ValidationConfig.from_env()  # Start with environment variables

        if self.config_path and Path(self.config_path).exists():
            # Override with file configuration
            file_config = ValidationConfig.from_yaml(self.config_path)
            config_dict = config.to_dict()
            config_dict.update(file_config.to_dict())
            config = ValidationConfig.from_dict(config_dict)

        # Validate configuration
        errors = config.validate()
        if errors:
            raise ValueError(f"Configuration validation failed: {', '.join(errors)}")

        return config

    def _should_reload(self) -> bool:
        """Check if configuration should be reloaded"""
        if not self.config_path:
            return False

        config_file = Path(self.config_path)
        if not config_file.exists():
            return False

        current_modified = datetime.fromtimestamp(config_file.stat().st_mtime)
        return self._last_modified is None or current_modified > self._last_modified

    def reload(self):
        """Force reload configuration"""
        self._config = None
        return self.get_config()

    def create_default_config(self, config_path: str):
        """Create default configuration file"""
        config = ValidationConfig()
        config.save(config_path)


# Global configuration instance
_default_config_manager = ConfigManager("validator/validation_config.yaml")


def get_config() -> ValidationConfig:
    """Get global configuration instance"""
    return _default_config_manager.get_config()


def set_config_path(config_path: str):
    """Set global configuration path"""
    global _default_config_manager
    _default_config_manager = ConfigManager(config_path)
