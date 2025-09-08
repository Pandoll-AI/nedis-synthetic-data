"""
설정 관리 모듈

NEDIS 합성 데이터 생성 시스템의 설정을 YAML 파일에서 로드하고 관리합니다.
런타임 설정 변경, 환경별 설정 오버라이드, 설정 검증 등을 지원합니다.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
import os
from datetime import datetime


class ConfigManager:
    """설정 관리자"""
    
    def __init__(self, config_path: str = "config/generation_params.yaml"):
        """
        설정 관리자 초기화
        
        Args:
            config_path: YAML 설정 파일 경로
        """
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """YAML 설정 파일 로드"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                self.logger.info(f"Configuration loaded from {self.config_path}")
                
                # 환경 변수로 오버라이드
                config = self._apply_env_overrides(config)
                
                # 설정 검증
                self._validate_config(config)
                
                return config
            else:
                self.logger.warning(f"Config file not found: {self.config_path}")
                return self._get_default_config()
                
        except yaml.YAMLError as e:
            self.logger.error(f"Invalid YAML in config file: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
            
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """환경 변수로 설정 오버라이드"""
        # NEDIS_DB_PATH 환경 변수가 있으면 database.db_path 오버라이드
        if os.getenv('NEDIS_DB_PATH'):
            config.setdefault('database', {})['db_path'] = os.getenv('NEDIS_DB_PATH')
            
        # NEDIS_LOG_LEVEL 환경 변수로 로그 레벨 오버라이드
        if os.getenv('NEDIS_LOG_LEVEL'):
            config.setdefault('logging', {})['level'] = os.getenv('NEDIS_LOG_LEVEL')
            
        # NEDIS_TARGET_RECORDS 환경 변수로 목표 레코드 수 오버라이드
        if os.getenv('NEDIS_TARGET_RECORDS'):
            try:
                target_records = int(os.getenv('NEDIS_TARGET_RECORDS'))
                config.setdefault('population', {})['target_total_records'] = target_records
            except ValueError:
                self.logger.warning("Invalid NEDIS_TARGET_RECORDS environment variable")
                
        return config
        
    def _validate_config(self, config: Dict[str, Any]):
        """설정 유효성 검증"""
        required_sections = ['population', 'temporal', 'allocation', 'clinical', 'validation']
        
        for section in required_sections:
            if section not in config:
                self.logger.warning(f"Missing config section: {section}")
                
        # 필수 파라미터 검증
        if 'population' in config:
            pop_config = config['population']
            if 'target_total_records' in pop_config:
                target = pop_config['target_total_records']
                if not isinstance(target, int) or target <= 0:
                    raise ValueError(f"Invalid target_total_records: {target}")
                    
        # 확률값 검증
        if 'allocation' in config and 'gravity_gamma' in config['allocation']:
            gamma = config['allocation']['gravity_gamma']
            if not isinstance(gamma, (int, float)) or gamma <= 0:
                raise ValueError(f"Invalid gravity_gamma: {gamma}")
            
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'database': {
                'db_path': 'nedis_synthetic.duckdb',
                'batch_size': 10000,
                'checkpoint_interval': 100000
            },
            'population': {
                'dirichlet_alpha': 1.0,
                'target_total_records': 9_200_000
            },
            'temporal': {
                'nhpp_seasonal_weights': {
                    'spring': 0.25, 'summer': 0.25,
                    'fall': 0.25, 'winter': 0.25
                },
                'holidays_2017': [
                    '20170101', '20170127', '20170128', '20170129', '20170130',
                    '20170301', '20170503', '20170506', '20170509', '20170815',
                    '20171003', '20171004', '20171005', '20171006', 
                    '20171009', '20171225'
                ]
            },
            'allocation': {
                'gravity_gamma': 1.5,
                'ipf_tolerance': 0.001,
                'ipf_max_iterations': 100,
                'capacity_buffer_multiplier': 2.0
            },
            'clinical': {
                'ktas_smoothing': 1.0,
                'diagnosis_min_count': 10,
                'bayesian_alpha': 1.0
            },
            'temporal_patterns': {
                'duration_sigma': 0.5,
                'vitals_measurement_rates': {
                    'ktas_1': 1.0, 'ktas_2': 0.95, 'ktas_3': 0.85,
                    'ktas_4': 0.70, 'ktas_5': 0.50
                }
            },
            'validation': {
                'statistical_alpha': 0.05,
                'privacy_k_anonymity': 5,
                'ks_test_threshold': 0.05,
                'chi2_test_threshold': 0.05,
                'correlation_threshold': 0.05
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_max_bytes': 10485760,
                'backup_count': 3
            },
            'optimization': {
                'bayesian_n_calls': 50,
                'objective_weights': {
                    'fidelity': 0.3, 'utility': 0.3,
                    'privacy': 0.2, 'clinical': 0.2
                }
            }
        }
        
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        점 표기법으로 중첩된 설정값 조회
        
        Args:
            key_path: 점으로 구분된 키 경로 (예: 'population.dirichlet_alpha')
            default: 기본값
            
        Returns:
            설정값 또는 기본값
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            else:
                self.logger.warning(f"Config key not found: {key_path}")
                return None
                
    def set(self, key_path: str, value: Any):
        """
        설정값 런타임 업데이트
        
        Args:
            key_path: 점으로 구분된 키 경로
            value: 새 값
        """
        keys = key_path.split('.')
        config = self.config
        
        # 중간 경로가 없으면 생성
        for key in keys[:-1]:
            config = config.setdefault(key, {})
            
        # 최종 값 설정
        config[keys[-1]] = value
        self.logger.info(f"Updated config: {key_path} = {value}")
        
    def get_phase_config(self, phase_name: str) -> Dict[str, Any]:
        """
        특정 Phase의 설정 반환
        
        Args:
            phase_name: Phase 이름 (population, temporal, allocation 등)
            
        Returns:
            Phase별 설정 딕셔너리
        """
        phase_config = self.get(phase_name, {})
        
        # 공통 설정도 포함
        common_config = {
            'database': self.get('database', {}),
            'logging': self.get('logging', {})
        }
        
        return {**common_config, phase_name: phase_config}
        
    def save_config(self, output_path: Optional[str] = None):
        """
        현재 설정을 YAML 파일로 저장
        
        Args:
            output_path: 저장할 파일 경로 (None이면 원본 파일에 저장)
        """
        save_path = Path(output_path) if output_path else self.config_path
        
        try:
            # 백업 생성
            if save_path.exists():
                backup_path = save_path.with_suffix(
                    f'.{datetime.now().strftime("%Y%m%d_%H%M%S")}.bak'
                )
                save_path.rename(backup_path)
                self.logger.info(f"Config backup created: {backup_path}")
                
            # 새 설정 저장
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
                
            self.logger.info(f"Configuration saved to: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            raise
            
    def validate_runtime_config(self) -> bool:
        """런타임 설정 유효성 검증"""
        try:
            # 데이터베이스 경로 검증
            db_path = self.get('database.db_path')
            if db_path:
                db_dir = Path(db_path).parent
                if not db_dir.exists():
                    self.logger.error(f"Database directory does not exist: {db_dir}")
                    return False
                    
            # 로그 디렉토리 검증
            logs_dir = Path('logs')
            if not logs_dir.exists():
                logs_dir.mkdir(parents=True)
                self.logger.info("Created logs directory")
                
            # 출력 디렉토리 검증
            outputs_dir = Path('outputs')
            if not outputs_dir.exists():
                outputs_dir.mkdir(parents=True)
                self.logger.info("Created outputs directory")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Runtime config validation failed: {e}")
            return False
            
    def get_summary(self) -> Dict[str, Any]:
        """설정 요약 정보 반환"""
        return {
            'config_file': str(self.config_path),
            'target_records': self.get('population.target_total_records'),
            'database_path': self.get('database.db_path'),
            'log_level': self.get('logging.level'),
            'phases_configured': list(self.config.keys()),
            'last_loaded': datetime.now().isoformat()
        }
        
    def __str__(self) -> str:
        """설정 정보 문자열 표현"""
        summary = self.get_summary()
        return f"ConfigManager(file={summary['config_file']}, target={summary['target_records']:,})"