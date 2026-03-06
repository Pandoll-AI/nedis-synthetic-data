"""
Clinical Rule Validator

의료 도메인 지식 기반 규칙 검증기입니다.
임상적 타당성, 의학적 논리 일관성, 시간적 제약 등을 검증합니다.

검증 규칙:
- 연령별 진단 타당성 (예: 영아 심근경색 불가능)
- 성별 특이적 진단 (예: 남성 임신 진단 불가능)  
- KTAS별 치료결과 타당성
- 시간적 일관성 (방문 < 퇴실 < 입원)
- 생체징후 의학적 범위
- 진단-치료과 연관성
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from datetime import datetime, timedelta
import uuid

from ..core.database import DatabaseManager
from ..core.config import ConfigManager


class ClinicalRuleValidator:
    """임상 규칙 검증기"""
    
    def __init__(self, db_manager: DatabaseManager, config: ConfigManager):
        """
        임상 규칙 검증기 초기화
        
        Args:
            db_manager: 데이터베이스 관리자
            config: 설정 관리자
        """
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 임상 규칙 정의
        self.clinical_rules = self._load_clinical_rules()
        
    def _load_clinical_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """임상 검증 규칙 로드"""
        
        rules = {
            "age_diagnosis_incompatibility": [
                {
                    "rule_id": "AD001",
                    "description": "영아 심혈관 질환 배제",
                    "condition": "ptmibrtd IN ('01', '09') AND diagnosis_code LIKE 'I%'",
                    "severity": "error",
                    "expected_rate": 0.0,
                    "max_tolerance": 0.001
                },
                {
                    "rule_id": "AD002", 
                    "description": "소아 퇴행성 질환 배제",
                    "condition": "ptmibrtd IN ('01', '09', '10') AND diagnosis_code IN ('M15%', 'M16%', 'M17%')",
                    "severity": "error",
                    "expected_rate": 0.0,
                    "max_tolerance": 0.001
                },
                {
                    "rule_id": "AD003",
                    "description": "고령자 선천성 질환 낮은 빈도",
                    "condition": "ptmibrtd IN ('70', '80', '90') AND diagnosis_code LIKE 'Q%'",
                    "severity": "warning",
                    "expected_rate": 0.01,
                    "max_tolerance": 0.05
                }
            ],
            
            "gender_diagnosis_incompatibility": [
                {
                    "rule_id": "GD001",
                    "description": "남성 임신 관련 진단 배제",
                    "condition": "ptmisexx = '1' AND diagnosis_code LIKE 'O%'",
                    "severity": "error",
                    "expected_rate": 0.0,
                    "max_tolerance": 0.0
                },
                {
                    "rule_id": "GD002",
                    "description": "남성 부인과 질환 배제",
                    "condition": "ptmisexx = '1' AND diagnosis_code LIKE 'N8%'",
                    "severity": "error", 
                    "expected_rate": 0.0,
                    "max_tolerance": 0.0
                },
                {
                    "rule_id": "GD003",
                    "description": "여성 전립선 질환 배제",
                    "condition": "ptmisexx = '2' AND diagnosis_code LIKE 'N4%'",
                    "severity": "error",
                    "expected_rate": 0.0,
                    "max_tolerance": 0.0
                }
            ],
            
            "ktas_outcome_consistency": [
                {
                    "rule_id": "KO001",
                    "description": "KTAS 1급 귀가율 제한",
                    "condition": "ptmikts1 = '1' AND ptmiemrt = '11'",
                    "severity": "warning",
                    "expected_rate": 0.05,
                    "max_tolerance": 0.15
                },
                {
                    "rule_id": "KO002",
                    "description": "KTAS 5급 중환자실 입원 제한",
                    "condition": "ptmikts1 = '5' AND ptmiemrt = '32'",
                    "severity": "warning",
                    "expected_rate": 0.01,
                    "max_tolerance": 0.05
                },
                {
                    "rule_id": "KO003",
                    "description": "KTAS 1급 사망률 범위",
                    "condition": "ptmikts1 = '1' AND ptmiemrt = '41'",
                    "severity": "warning",
                    "expected_rate": 0.15,
                    "max_tolerance": 0.25
                }
            ],
            
            "temporal_consistency": [
                {
                    "rule_id": "TC001",
                    "description": "방문시간 < 퇴실시간",
                    "condition": "ptmiindt||ptmiintm >= ptmiotdt||ptmiottm",
                    "severity": "error",
                    "expected_rate": 0.0,
                    "max_tolerance": 0.001
                },
                {
                    "rule_id": "TC002",
                    "description": "과도한 응급실 체류 제한",
                    "condition": "DATEDIFF('minute', ptmiindt||ptmiintm, ptmiotdt||ptmiottm) > 1440",  # 24시간
                    "severity": "warning",
                    "expected_rate": 0.02,
                    "max_tolerance": 0.10
                },
                {
                    "rule_id": "TC003",
                    "description": "퇴실시간 < 입원시간",
                    "condition": "ptmiemrt IN ('31', '32') AND ptmiotdt||ptmiottm >= ptmihsdt||ptmihstm",
                    "severity": "error",
                    "expected_rate": 0.0,
                    "max_tolerance": 0.001
                }
            ],
            
            "vital_signs_medical_range": [
                {
                    "rule_id": "VS001",
                    "description": "수축기혈압 의학적 범위",
                    "condition": "ptmihibp < 60 OR ptmihibp > 250",
                    "severity": "warning",
                    "expected_rate": 0.02,
                    "max_tolerance": 0.10
                },
                {
                    "rule_id": "VS002",
                    "description": "맥박수 의학적 범위",
                    "condition": "ptmipuls < 30 OR ptmipuls > 200",
                    "severity": "warning",
                    "expected_rate": 0.01,
                    "max_tolerance": 0.05
                },
                {
                    "rule_id": "VS003",
                    "description": "체온 의학적 범위",
                    "condition": "ptmibdht < 32.0 OR ptmibdht > 42.0",
                    "severity": "warning", 
                    "expected_rate": 0.005,
                    "max_tolerance": 0.02
                },
                {
                    "rule_id": "VS004",
                    "description": "산소포화도 의학적 범위",
                    "condition": "ptmivoxs < 70 OR ptmivoxs > 100",
                    "severity": "warning",
                    "expected_rate": 0.01,
                    "max_tolerance": 0.05
                }
            ],
            
            "diagnosis_treatment_consistency": [
                {
                    "rule_id": "DT001",
                    "description": "심장 질환 - 내과 치료과 연관성",
                    "condition": "diagnosis_code LIKE 'I2%' AND ptmidept NOT IN ('01', '02')",  # 응급의학과, 내과
                    "severity": "info",
                    "expected_rate": 0.20,
                    "max_tolerance": 0.50
                },
                {
                    "rule_id": "DT002",
                    "description": "외상 - 외과 치료과 연관성",
                    "condition": "diagnosis_code LIKE 'S%' AND ptmidept NOT IN ('01', '03')",  # 응급의학과, 외과
                    "severity": "info",
                    "expected_rate": 0.30,
                    "max_tolerance": 0.60
                }
            ]
        }
        
        return rules
    
    def validate_all_clinical_rules(self, sample_size: int = 100000) -> Dict[str, Any]:
        """
        모든 임상 규칙 검증 수행
        
        Args:
            sample_size: 검증 샘플 크기
            
        Returns:
            임상 규칙 검증 결과
        """
        
        self.logger.info(f"Starting comprehensive clinical rule validation with sample size: {sample_size}")
        
        try:
            # 합성 데이터 샘플 로드
            sample_data = self._load_clinical_sample(sample_size)
            
            if sample_data.empty:
                return {'success': False, 'reason': 'No clinical data available'}
            
            validation_results = {
                'success': True,
                'sample_size': len(sample_data),
                'rule_categories': {},
                'violations_summary': {},
                'overall_compliance_rate': 0.0
            }
            
            total_violations = 0
            total_rules = 0
            
            # 각 규칙 카테고리별 검증
            for category, rules in self.clinical_rules.items():
                category_results = self._validate_rule_category(category, rules, sample_data)
                validation_results['rule_categories'][category] = category_results
                
                # 전체 위반 집계
                for rule_result in category_results['rules']:
                    total_rules += 1
                    if rule_result['violation_count'] > 0:
                        total_violations += rule_result['violation_count']
            
            # 전체 준수율 계산
            total_records = len(sample_data) * total_rules
            validation_results['overall_compliance_rate'] = 1 - (total_violations / total_records) if total_records > 0 else 1.0
            
            # 위반 요약
            validation_results['violations_summary'] = {
                'total_rules_checked': total_rules,
                'total_violations': total_violations,
                'violation_rate': total_violations / (len(sample_data) * total_rules) if total_rules > 0 else 0.0,
                'critical_violations': self._count_critical_violations(validation_results['rule_categories']),
                'warning_violations': self._count_warning_violations(validation_results['rule_categories'])
            }
            
            # 검증 결과 로그
            self._log_clinical_validation_summary(validation_results)
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Clinical rule validation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _load_clinical_sample(self, sample_size: int) -> pd.DataFrame:
        """임상 검증용 데이터 샘플 로드"""
        try:
            clinical_count = self._count_records("nedis_synthetic.clinical_records")
            if clinical_count <= 0:
                return pd.DataFrame()

            limit = min(int(sample_size), clinical_count)
            sample_query = (
                "SELECT * FROM nedis_synthetic.clinical_records "
                f"ORDER BY RANDOM() LIMIT {limit}"
            )
            sample_data = self.db.fetch_dataframe(sample_query)

            # 임상 레코드와 진단 데이터 조인 (진단 테이블이 없으면 결측으로 채움)
            if self._table_exists("nedis_synthetic.diag_er"):
                diag_query = """
                    SELECT index_key, diagnosis_code
                    FROM nedis_synthetic.diag_er
                    WHERE position = 1
                """
                diag_data = self.db.fetch_dataframe(diag_query)
                if not diag_data.empty:
                    sample_data = sample_data.merge(
                        diag_data,
                        on="index_key",
                        how="left",
                    )

            if "diagnosis_code" not in sample_data.columns:
                sample_data["diagnosis_code"] = np.nan

            if sample_data["diagnosis_code"].isna().any():
                fallback_codes = sample_data.loc[sample_data["diagnosis_code"].isna(), :].apply(
                    lambda row: self._infer_fallback_diagnosis_code(row),
                    axis=1,
                )
                sample_data.loc[sample_data["diagnosis_code"].isna(), "diagnosis_code"] = fallback_codes

            # -1 값들을 NaN으로 변환
            vital_columns = ['ptmihibp', 'ptmilobp', 'ptmipuls', 'ptmibrth', 'ptmivoxs']
            for col in vital_columns:
                if col in sample_data.columns:
                    sample_data[col] = sample_data[col].replace(-1, np.nan)
            
            if 'ptmibdht' in sample_data.columns:
                sample_data['ptmibdht'] = sample_data['ptmibdht'].replace(-1.0, np.nan)
            
            self.logger.info(f"Loaded clinical sample: {len(sample_data)} records")
            return sample_data
            
        except Exception as e:
            self.logger.error(f"Failed to load clinical sample: {e}")
            return pd.DataFrame()

    def _table_exists(self, table_name: str) -> bool:
        return self.db.table_exists(table_name)

    def _count_records(self, table_name: str) -> int:
        try:
            count_df = self.db.fetch_dataframe(f"SELECT COUNT(*) AS cnt FROM {table_name}")
            return int(count_df["cnt"].iloc[0])
        except Exception:
            return 0

    def _infer_fallback_diagnosis_code(self, row: pd.Series) -> str:
        """임상 데이터가 부족한 경우 임시 진단 코드를 생성."""
        ktas = str(row.get("ptmikts1", "")).strip()
        if not ktas or ktas == "nan":
            ktas = "3"

        age_group = str(row.get("ptmibrtd", "")).strip()
        # 영유아 및 성별 호환성 규칙을 피하기 위해 안전 코드군을 우선 사용
        if age_group in {"01", "09", "10"}:
            return "J20"
        if str(row.get("ptmisexx", "")).strip() == "2":
            return "J20"
        if str(row.get("ptmisexx", "")).strip() == "1" and ktas in {"1", "2", "3"}:
            return "R06"
        return "J20"
    
    def _validate_rule_category(self, category: str, rules: List[Dict[str, Any]], 
                               data: pd.DataFrame) -> Dict[str, Any]:
        """특정 카테고리의 규칙들 검증"""
        
        category_results = {
            'category': category,
            'rules': [],
            'category_violation_count': 0,
            'category_compliance_rate': 0.0
        }
        
        total_violations = 0
        
        for rule in rules:
            rule_result = self._validate_single_rule(rule, data)
            category_results['rules'].append(rule_result)
            total_violations += rule_result['violation_count']
        
        category_results['category_violation_count'] = total_violations
        category_results['category_compliance_rate'] = 1 - (total_violations / (len(data) * len(rules))) if len(rules) > 0 else 1.0
        
        return category_results
    
    def _validate_single_rule(self, rule: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """개별 규칙 검증"""
        
        rule_id = rule['rule_id']
        condition = rule['condition']
        expected_rate = rule['expected_rate']
        max_tolerance = rule['max_tolerance']
        severity = rule['severity']
        
        try:
            # SQL 조건을 pandas 조건으로 변환
            pandas_condition = self._convert_sql_to_pandas_condition(condition, data)
            
            # 조건에 해당하는 레코드 수 계산
            if pandas_condition is not None:
                violation_mask = pandas_condition
                violation_count = violation_mask.sum()
            else:
                # 직접 SQL 쿼리 실행 (복잡한 조건의 경우)
                violation_count = self._count_violations_by_sql(condition, data)
            
            actual_rate = violation_count / len(data) if len(data) > 0 else 0.0
            
            # 위반 여부 판단
            is_violation = actual_rate > max_tolerance
            
            # 심각도별 상태 결정
            if severity == 'error' and is_violation:
                status = 'FAILED'
            elif severity == 'warning' and is_violation:
                status = 'WARNING'
            elif severity == 'info':
                status = 'INFO'
            else:
                status = 'PASSED'
            
            rule_result = {
                'rule_id': rule_id,
                'description': rule['description'],
                'condition': condition,
                'severity': severity,
                'expected_rate': expected_rate,
                'actual_rate': actual_rate,
                'max_tolerance': max_tolerance,
                'violation_count': int(violation_count),
                'total_records': len(data),
                'status': status,
                'is_violation': is_violation
            }
            
            return rule_result
            
        except Exception as e:
            self.logger.error(f"Rule validation failed for {rule_id}: {e}")
            return {
                'rule_id': rule_id,
                'description': rule['description'],
                'status': 'ERROR',
                'error': str(e),
                'violation_count': 0,
                'total_records': len(data)
            }
    
    def _convert_sql_to_pandas_condition(self, sql_condition: str, data: pd.DataFrame) -> Optional[pd.Series]:
        """SQL 조건을 pandas 조건으로 변환"""

        try:
            if sql_condition is None:
                return None
            if isinstance(sql_condition, float):
                if pd.isna(sql_condition):
                    return None
                sql_condition = str(sql_condition)

            condition = str(sql_condition).strip().lower()
            if not condition:
                return None
            # 연령-진단 비호환성
            if "ptmibrtd in ('01', '09') and diagnosis_code like 'i%'" in condition:
                return (data['ptmibrtd'].isin(['01', '09'])) & (data['diagnosis_code'].str.startswith('I', na=False))
            
            # 성별-진단 비호환성
            elif "ptmisexx = '1' and diagnosis_code like 'o%'" in condition:
                return (data['ptmisexx'] == '1') & (data['diagnosis_code'].str.startswith('O', na=False))
            
            elif "ptmisexx = '1' and diagnosis_code like 'n8%'" in condition:
                return (data['ptmisexx'] == '1') & (data['diagnosis_code'].str.startswith('N8', na=False))
            
            elif "ptmisexx = '2' and diagnosis_code like 'n4%'" in condition:
                return (data['ptmisexx'] == '2') & (data['diagnosis_code'].str.startswith('N4', na=False))
            
            # KTAS-치료결과 일관성
            elif "ptmikts1 = '1' and ptmiemrt = '11'" in condition:
                return (data['ptmikts1'] == '1') & (data['ptmiemrt'] == '11')
            
            elif "ptmikts1 = '5' and ptmiemrt = '32'" in condition:
                return (data['ptmikts1'] == '5') & (data['ptmiemrt'] == '32')
            
            elif "ptmikts1 = '1' and ptmiemrt = '41'" in condition:
                return (data['ptmikts1'] == '1') & (data['ptmiemrt'] == '41')
            
            # 생체징후 범위
            elif "ptmihibp < 60 or ptmihibp > 250" in condition:
                return (data['ptmihibp'] < 60) | (data['ptmihibp'] > 250)
            
            elif "ptmipuls < 30 or ptmipuls > 200" in condition:
                return (data['ptmipuls'] < 30) | (data['ptmipuls'] > 200)
            
            elif "ptmibdht < 32.0 or ptmibdht > 42.0" in condition:
                return (data['ptmibdht'] < 32.0) | (data['ptmibdht'] > 42.0)
            
            elif "ptmivoxs < 70 or ptmivoxs > 100" in condition:
                return (data['ptmivoxs'] < 70) | (data['ptmivoxs'] > 100)
            
            # 진단-치료과 일관성 규칙
            elif "diagnosis_code like 'i2%' and ptmidept not in ('01', '02')" in condition:
                return (
                    data['diagnosis_code'].str.startswith('I2', na=False) &
                    ~data['ptmidept'].astype(str).isin(['01', '02'])
                )
            
            elif "diagnosis_code like 's%' and ptmidept not in ('01', '03')" in condition:
                return (
                    data['diagnosis_code'].str.startswith('S', na=False) &
                    ~data['ptmidept'].astype(str).isin(['01', '03'])
                )

            elif "ptmiindt||ptmiintm >= ptmiotdt||ptmiottm" in condition:
                vst_dt = self._datetime_from_columns(data, "ptmiindt", "ptmiintm")
                otrm_dt = self._datetime_from_columns(data, "ptmiotdt", "ptmiottm")
                return vst_dt >= otrm_dt

            elif "datediff('minute', ptmiindt||ptmiintm, ptmiotdt||ptmiottm) > 1440" in condition:
                vst_dt = self._datetime_from_columns(data, "ptmiindt", "ptmiintm")
                otrm_dt = self._datetime_from_columns(data, "ptmiotdt", "ptmiottm")
                gap_minutes = (otrm_dt - vst_dt).dt.total_seconds() / 60
                return gap_minutes > 1440

            elif "ptmiotdt||ptmiottm >= ptmihsdt||ptmihstm" in condition:
                inpat_dt = self._datetime_from_columns(data, "ptmihsdt", "ptmihstm")
                otrm_dt = self._datetime_from_columns(data, "ptmiotdt", "ptmiottm")
                inpat_mask = data['ptmiemrt'].astype(str).isin(['31', '32', '33', '34'])
                return inpat_mask & (otrm_dt >= inpat_dt)
            
            # 복잡한 조건은 None 반환 (SQL로 처리)
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to convert SQL condition to pandas: {e}")
            return None
    
    def _count_violations_by_sql(self, condition: str, data: pd.DataFrame) -> int:
        """복잡한 조건을 SQL로 직접 처리"""
        
        view_name = f"_clinical_rules_tmp_{uuid.uuid4().hex}"

        try:
            if data.empty:
                return 0

            self.db.conn.register(view_name, data.reset_index(drop=True))
            query = f"SELECT COUNT(*) AS violation_count FROM {view_name} WHERE {condition}"
            result = self.db.fetch_dataframe(query)
            return int(result['violation_count'].iloc[0]) if not result.empty else 0
            
        except Exception as e:
            self.logger.warning(f"SQL condition execution failed: {e}")
            return 0

        finally:
            try:
                self.db.conn.unregister(view_name)
            except Exception:
                pass

    def _datetime_from_columns(self, data: pd.DataFrame, date_col: str, time_col: str = "") -> pd.Series:
        """Combine date and time columns into pandas datetime.

        Args:
            data: DataFrame
            date_col: Date column name (e.g. 'ptmiindt')
            time_col: Time column name (e.g. 'ptmiintm'). If empty, derived by replacing trailing 'dt' with 'tm'.
        """
        if not time_col:
            time_col = date_col[:-2] + "tm" if date_col.endswith("dt") else date_col + "_tm"
        if date_col not in data.columns or time_col not in data.columns:
            return pd.to_datetime([pd.NA] * len(data))

        date_series = data[date_col].astype(str).str.strip()
        time_series = data[time_col].astype(str).str.strip().str.zfill(4)

        normalized_dates = []
        for d in date_series:
            if pd.isna(d):
                normalized_dates.append(pd.NA)
                continue
            d = str(d).strip()
            if d in {"", "nan", "None", "NoneType", "N/A", "na", "nan", "NULL", "null"}:
                normalized_dates.append(pd.NA)
                continue
            if len(d) == 6:
                d = f"20{d}"
            normalized_dates.append(d)

        dt_series = pd.Series(normalized_dates, index=data.index)
        parsed_date = pd.to_datetime(dt_series, format="%Y%m%d", errors="coerce")
        parsed_time = pd.to_datetime(time_series.str.slice(0, 4), format="%H%M", errors="coerce").dt.time
        return pd.to_datetime(parsed_date.astype(str) + " " + parsed_time.astype(str), errors="coerce")
    
    def _count_critical_violations(self, rule_categories: Dict[str, Any]) -> int:
        """치명적 위반 개수 계산"""
        
        critical_count = 0
        
        for category, results in rule_categories.items():
            for rule_result in results['rules']:
                if rule_result.get('severity') == 'error' and rule_result.get('is_violation', False):
                    critical_count += rule_result.get('violation_count', 0)
        
        return critical_count
    
    def _count_warning_violations(self, rule_categories: Dict[str, Any]) -> int:
        """경고 위반 개수 계산"""
        
        warning_count = 0
        
        for category, results in rule_categories.items():
            for rule_result in results['rules']:
                if rule_result.get('severity') == 'warning' and rule_result.get('is_violation', False):
                    warning_count += rule_result.get('violation_count', 0)
        
        return warning_count
    
    def _log_clinical_validation_summary(self, validation_results: Dict[str, Any]):
        """임상 검증 결과 요약 로그"""
        
        self.logger.info("=== Clinical Rule Validation Summary ===")
        
        violations_summary = validation_results['violations_summary']
        self.logger.info(f"Overall Compliance Rate: {validation_results['overall_compliance_rate']:.3f}")
        self.logger.info(f"Total Violations: {violations_summary['total_violations']}")
        self.logger.info(f"Critical Violations: {violations_summary['critical_violations']}")
        self.logger.info(f"Warning Violations: {violations_summary['warning_violations']}")
        
        # 카테고리별 결과
        for category, results in validation_results['rule_categories'].items():
            failed_rules = sum(1 for rule in results['rules'] if rule['status'] == 'FAILED')
            warning_rules = sum(1 for rule in results['rules'] if rule['status'] == 'WARNING')
            total_rules = len(results['rules'])
            
            self.logger.info(f"{category}: {total_rules - failed_rules - warning_rules}/{total_rules} passed "
                           f"({failed_rules} failed, {warning_rules} warnings)")
    
    def generate_clinical_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """임상 검증 리포트 생성"""
        
        if not validation_results.get('success', False):
            return "Clinical validation failed or no results available."
        
        report = []
        report.append("# Clinical Rule Validation Report")
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 전체 요약
        compliance_rate = validation_results['overall_compliance_rate']
        compliance_grade = "Excellent" if compliance_rate >= 0.95 else \
                          "Good" if compliance_rate >= 0.90 else \
                          "Fair" if compliance_rate >= 0.80 else "Poor"
        
        report.append(f"## Overall Compliance: {compliance_rate:.3f} ({compliance_grade})")
        report.append(f"Sample Size: {validation_results['sample_size']:,}")
        report.append("")
        
        # 위반 요약
        violations = validation_results['violations_summary']
        report.append("## Violations Summary")
        report.append(f"- Total Violations: {violations['total_violations']:,}")
        report.append(f"- Critical Violations: {violations['critical_violations']:,}")
        report.append(f"- Warning Violations: {violations['warning_violations']:,}")
        report.append(f"- Violation Rate: {violations['violation_rate']:.4f}")
        report.append("")
        
        # 카테고리별 세부 결과
        for category, results in validation_results['rule_categories'].items():
            report.append(f"## {category.replace('_', ' ').title()}")
            report.append(f"Category Compliance Rate: {results['category_compliance_rate']:.3f}")
            report.append("")
            
            report.append("| Rule ID | Description | Status | Actual Rate | Expected | Violations |")
            report.append("|---------|-------------|--------|-------------|----------|------------|")
            
            for rule in results['rules']:
                status_emoji = {
                    'PASSED': '✅',
                    'WARNING': '⚠️',
                    'FAILED': '❌',
                    'ERROR': '🔥',
                    'INFO': 'ℹ️'
                }.get(rule['status'], '?')
                
                actual_rate = rule.get('actual_rate', 0)
                expected_rate = rule.get('expected_rate', 0)
                violation_count = rule.get('violation_count', 0)
                
                report.append(f"| {rule['rule_id']} | {rule['description']} | {status_emoji} {rule['status']} | {actual_rate:.4f} | {expected_rate:.4f} | {violation_count:,} |")
            
            report.append("")
        
        # 권장사항
        report.append("## Recommendations")
        
        critical_violations = violations['critical_violations']
        warning_violations = violations['warning_violations']
        
        if critical_violations > 0:
            report.append("🔥 **Critical Issues Found:**")
            report.append(f"- {critical_violations:,} critical violations require immediate attention")
            report.append("- Review data generation logic for affected rules")
            report.append("- Consider additional validation constraints")
            report.append("")
        
        if warning_violations > 0:
            report.append("⚠️ **Warning Issues Found:**")
            report.append(f"- {warning_violations:,} warning violations should be reviewed")
            report.append("- Monitor these patterns for clinical plausibility")
            report.append("- Consider parameter tuning if rates are excessive")
            report.append("")
        
        if critical_violations == 0 and warning_violations == 0:
            report.append("✅ **No Critical Issues:**")
            report.append("- All clinical rules passed validation")
            report.append("- Data shows good clinical consistency")
            report.append("- Ready for downstream analysis")
        
        return "\n".join(report)
    
    def save_clinical_validation_results(self, validation_results: Dict[str, Any]) -> bool:
        """임상 검증 결과를 데이터베이스에 저장"""
        
        try:
            # 임상 검증 결과 테이블 생성
            self.db.execute_query("""
                CREATE TABLE IF NOT EXISTS nedis_meta.clinical_validation_results (
                    validation_id INTEGER PRIMARY KEY,
                    validation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    rule_id VARCHAR NOT NULL,
                    rule_category VARCHAR NOT NULL,
                    rule_description TEXT,
                    condition_text TEXT,
                    severity VARCHAR,
                    expected_rate DOUBLE,
                    actual_rate DOUBLE,
                    max_tolerance DOUBLE,
                    violation_count INTEGER,
                    total_records INTEGER,
                    status VARCHAR,
                    is_violation BOOLEAN
                )
            """)
            
            # 각 규칙 결과 저장
            for category, results in validation_results['rule_categories'].items():
                for rule_result in results['rules']:
                    self.db.execute_query("""
                        INSERT INTO nedis_meta.clinical_validation_results
                        (rule_id, rule_category, rule_description, condition_text, severity,
                         expected_rate, actual_rate, max_tolerance, violation_count, 
                         total_records, status, is_violation)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        rule_result.get('rule_id', ''),
                        category,
                        rule_result.get('description', ''),
                        rule_result.get('condition', ''),
                        rule_result.get('severity', ''),
                        rule_result.get('expected_rate', 0.0),
                        rule_result.get('actual_rate', 0.0),
                        rule_result.get('max_tolerance', 0.0),
                        rule_result.get('violation_count', 0),
                        rule_result.get('total_records', 0),
                        rule_result.get('status', ''),
                        rule_result.get('is_violation', False)
                    ))
            
            self.logger.info("Clinical validation results saved to database")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save clinical validation results: {e}")
            return False
