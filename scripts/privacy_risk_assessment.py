#!/usr/bin/env python3
"""
Privacy Risk Assessment for NEDIS Synthetic Data
냉정하고 객관적인 재식별 위험 평가
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
import hashlib
import json
from typing import Dict, List, Tuple, Any

from src.core.database import DatabaseManager
from src.core.config import ConfigManager
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class PrivacyRiskAssessment:
    """
    NEDIS 합성 데이터의 재식별 위험을 객관적으로 평가
    """
    
    def __init__(self, original_db='nedis_data.duckdb', synthetic_db='nedis_synthetic.duckdb'):
        self.original_db = DatabaseManager(original_db)
        self.synthetic_db = DatabaseManager(synthetic_db) if Path(synthetic_db).exists() else None
        self.risks = {
            'direct_identifiers': {},
            'quasi_identifiers': {},
            'k_anonymity': {},
            'l_diversity': {},
            'linkage_attacks': {},
            'temporal_patterns': {},
            'overall_risk': None
        }
        
    def assess_all_risks(self, sample_size=10000):
        """모든 재식별 위험 요소를 평가"""
        logger.info("=" * 80)
        logger.info("NEDIS 합성 데이터 재식별 위험 평가 시작")
        logger.info("=" * 80)
        
        # 데이터 로드
        logger.info(f"\n1. 데이터 로드 (샘플 크기: {sample_size:,})")
        orig_data = self._load_original_data(sample_size)
        synth_data = self._load_or_generate_synthetic_data(orig_data, sample_size)
        
        # 1. 직접 식별자 검사
        logger.info("\n2. 직접 식별자 검사")
        self.risks['direct_identifiers'] = self._check_direct_identifiers(orig_data, synth_data)
        
        # 2. 준식별자 조합 분석
        logger.info("\n3. 준식별자 조합 분석")
        self.risks['quasi_identifiers'] = self._analyze_quasi_identifiers(orig_data, synth_data)
        
        # 3. k-익명성 평가
        logger.info("\n4. k-익명성 평가")
        self.risks['k_anonymity'] = self._evaluate_k_anonymity(synth_data)
        
        # 4. l-다양성 평가
        logger.info("\n5. l-다양성 평가")
        self.risks['l_diversity'] = self._evaluate_l_diversity(synth_data)
        
        # 5. 연결 공격 취약성
        logger.info("\n6. 연결 공격 취약성 분석")
        self.risks['linkage_attacks'] = self._assess_linkage_attacks(orig_data, synth_data)
        
        # 6. 시간 패턴 재식별 위험
        logger.info("\n7. 시간 패턴 재식별 위험")
        self.risks['temporal_patterns'] = self._assess_temporal_patterns(orig_data, synth_data)
        
        # 7. 종합 위험도 계산
        logger.info("\n8. 종합 위험도 계산")
        self.risks['overall_risk'] = self._calculate_overall_risk()
        
        # 보고서 생성
        self._generate_report()
        
        return self.risks
    
    def _load_original_data(self, sample_size):
        """원본 데이터 로드"""
        query = f"""
        SELECT *
        FROM nedis_data.nedis2017
        LIMIT {sample_size}
        """
        return self.original_db.fetch_dataframe(query)
    
    def _load_or_generate_synthetic_data(self, orig_data, sample_size):
        """합성 데이터 로드 또는 생성"""
        # 실제 합성 프로세스를 시뮬레이션
        synth_data = orig_data.copy()
        
        # 직접 식별자 제거 시뮬레이션
        if 'pat_reg_no' in synth_data.columns:
            synth_data['pat_reg_no'] = np.random.randint(1000000, 9999999, len(synth_data))
        
        # 노이즈 추가
        numerical_cols = ['pat_age', 'vst_sbp', 'vst_dbp', 'vst_per_pu']
        for col in numerical_cols:
            if col in synth_data.columns:
                numeric_col = pd.to_numeric(synth_data[col], errors='coerce')
                valid_mask = numeric_col.notna()
                noise = np.random.normal(0, numeric_col[valid_mask].std() * 0.1, valid_mask.sum())
                synth_data.loc[valid_mask, col] = numeric_col[valid_mask] + noise
        
        return synth_data
    
    def _check_direct_identifiers(self, orig_data, synth_data):
        """직접 식별자 검사"""
        risks = {
            'found_identifiers': [],
            'risk_level': 'LOW'
        }
        
        # 직접 식별자 목록
        direct_identifiers = [
            'pat_reg_no',  # 환자 등록 번호
            'index_key',   # 인덱스 키
            'pat_brdt'     # 생년월일
        ]
        
        for identifier in direct_identifiers:
            if identifier in synth_data.columns:
                # 원본과 합성 데이터의 값 비교
                if identifier in orig_data.columns:
                    overlap = set(orig_data[identifier]) & set(synth_data[identifier])
                    overlap_pct = len(overlap) / len(orig_data[identifier]) * 100
                    
                    if overlap_pct > 0:
                        risks['found_identifiers'].append({
                            'column': identifier,
                            'overlap_percentage': overlap_pct,
                            'risk': 'CRITICAL' if overlap_pct > 50 else 'HIGH'
                        })
                        risks['risk_level'] = 'CRITICAL'
        
        logger.info(f"   - 직접 식별자 위험: {risks['risk_level']}")
        if risks['found_identifiers']:
            for id_risk in risks['found_identifiers']:
                logger.warning(f"   ⚠️  {id_risk['column']}: {id_risk['overlap_percentage']:.1f}% 중복")
        
        return risks
    
    def _analyze_quasi_identifiers(self, orig_data, synth_data):
        """준식별자 조합 분석"""
        risks = {
            'unique_combinations': {},
            'risk_level': 'MEDIUM'
        }
        
        # 준식별자 조합들
        quasi_id_sets = [
            ['pat_age', 'pat_sex', 'pat_sarea'],  # 나이+성별+지역
            ['ktas01', 'pat_age', 'pat_sex'],      # KTAS+나이+성별
            ['vst_dt', 'pat_sarea', 'ktas01']      # 방문날짜+지역+KTAS
        ]
        
        for qi_set in quasi_id_sets:
            if all(col in synth_data.columns for col in qi_set):
                # 조합 생성
                synth_combinations = synth_data[qi_set].fillna('NA').astype(str).agg('-'.join, axis=1)
                
                # 유일한 조합 찾기
                combo_counts = synth_combinations.value_counts()
                unique_combos = combo_counts[combo_counts == 1]
                
                unique_pct = len(unique_combos) / len(synth_data) * 100
                
                risks['unique_combinations']['-'.join(qi_set)] = {
                    'unique_records': len(unique_combos),
                    'percentage': unique_pct,
                    'risk': 'HIGH' if unique_pct > 10 else 'MEDIUM' if unique_pct > 5 else 'LOW'
                }
                
                if unique_pct > 10:
                    risks['risk_level'] = 'HIGH'
                
                logger.info(f"   - {'-'.join(qi_set)}: {unique_pct:.1f}% 유일")
        
        return risks
    
    def _evaluate_k_anonymity(self, synth_data):
        """k-익명성 평가"""
        results = {
            'minimum_k': None,
            'average_k': None,
            'k_distribution': {},
            'risk_level': 'MEDIUM'
        }
        
        # 주요 준식별자 조합
        qi_columns = ['pat_age', 'pat_sex', 'pat_sarea', 'ktas01']
        available_qi = [col for col in qi_columns if col in synth_data.columns]
        
        if len(available_qi) >= 3:
            # 준식별자 조합 생성
            qi_groups = synth_data[available_qi].fillna('NA').astype(str).agg('-'.join, axis=1)
            group_sizes = qi_groups.value_counts()
            
            results['minimum_k'] = int(group_sizes.min())
            results['average_k'] = float(group_sizes.mean())
            
            # k 분포
            for k_threshold in [1, 2, 3, 5, 10]:
                pct_below = (group_sizes < k_threshold).sum() / len(group_sizes) * 100
                results['k_distribution'][f'k<{k_threshold}'] = pct_below
            
            # 위험도 평가
            if results['minimum_k'] < 3:
                results['risk_level'] = 'HIGH'
            elif results['minimum_k'] < 5:
                results['risk_level'] = 'MEDIUM'
            else:
                results['risk_level'] = 'LOW'
            
            logger.info(f"   - 최소 k: {results['minimum_k']}")
            logger.info(f"   - 평균 k: {results['average_k']:.1f}")
            logger.info(f"   - k<5 비율: {results['k_distribution'].get('k<5', 0):.1f}%")
        
        return results
    
    def _evaluate_l_diversity(self, synth_data):
        """l-다양성 평가"""
        results = {
            'minimum_l': None,
            'average_l': None,
            'risk_level': 'MEDIUM'
        }
        
        # 민감 속성 (예: 진단, 증상)
        sensitive_attr = 'msypt'  # 주증상
        qi_columns = ['pat_age', 'pat_sex', 'pat_sarea']
        
        if sensitive_attr in synth_data.columns and all(col in synth_data.columns for col in qi_columns):
            # 준식별자 그룹별 민감속성 다양성
            qi_groups = synth_data[qi_columns].fillna('NA').astype(str).agg('-'.join, axis=1)
            
            l_values = []
            for group_id in qi_groups.unique():
                group_mask = qi_groups == group_id
                sensitive_values = synth_data.loc[group_mask, sensitive_attr].nunique()
                l_values.append(sensitive_values)
            
            if l_values:
                results['minimum_l'] = int(min(l_values))
                results['average_l'] = float(np.mean(l_values))
                
                if results['minimum_l'] < 2:
                    results['risk_level'] = 'HIGH'
                elif results['minimum_l'] < 3:
                    results['risk_level'] = 'MEDIUM'
                else:
                    results['risk_level'] = 'LOW'
                
                logger.info(f"   - 최소 l-다양성: {results['minimum_l']}")
                logger.info(f"   - 평균 l-다양성: {results['average_l']:.1f}")
        
        return results
    
    def _assess_linkage_attacks(self, orig_data, synth_data):
        """연결 공격 취약성 평가"""
        results = {
            'exact_matches': 0,
            'fuzzy_matches': 0,
            'vulnerable_records': [],
            'risk_level': 'MEDIUM'
        }
        
        # 희귀 조합 찾기 (KTAS 1 + 특정 증상)
        if 'ktas01' in synth_data.columns and 'msypt' in synth_data.columns:
            # KTAS 1 환자들
            ktas1_synth = synth_data[synth_data['ktas01'] == 1]
            ktas1_orig = orig_data[orig_data['ktas01'] == 1] if 'ktas01' in orig_data.columns else pd.DataFrame()
            
            if len(ktas1_synth) > 0 and len(ktas1_orig) > 0:
                # 희귀 조합 체크
                for idx, synth_row in ktas1_synth.iterrows():
                    # 매칭 가능한 원본 레코드 찾기
                    matches = 0
                    if 'pat_age' in orig_data.columns and 'pat_sex' in orig_data.columns:
                        # Convert to numeric for comparison
                        orig_age = pd.to_numeric(orig_data['pat_age'], errors='coerce')
                        synth_age = pd.to_numeric(synth_row['pat_age'], errors='coerce')
                        
                        if pd.notna(synth_age):
                            age_match = abs(orig_age - synth_age) <= 2
                            sex_match = orig_data['pat_sex'] == synth_row['pat_sex']
                            ktas_match = orig_data['ktas01'] == synth_row['ktas01']
                            
                            potential_matches = orig_data[age_match & sex_match & ktas_match]
                            matches = len(potential_matches)
                    
                    if matches == 1:
                        results['exact_matches'] += 1
                        results['vulnerable_records'].append(idx)
                    elif matches <= 3:
                        results['fuzzy_matches'] += 1
                
                # 위험도 평가
                match_rate = (results['exact_matches'] + results['fuzzy_matches']) / len(synth_data) * 100
                if match_rate > 5:
                    results['risk_level'] = 'HIGH'
                elif match_rate > 2:
                    results['risk_level'] = 'MEDIUM'
                else:
                    results['risk_level'] = 'LOW'
                
                logger.info(f"   - 정확한 매칭: {results['exact_matches']} 레코드")
                logger.info(f"   - 퍼지 매칭: {results['fuzzy_matches']} 레코드")
        
        return results
    
    def _assess_temporal_patterns(self, orig_data, synth_data):
        """시간 패턴 재식별 위험 평가"""
        results = {
            'unique_temporal_patterns': 0,
            'repeated_visits': 0,
            'risk_level': 'MEDIUM'
        }
        
        # 시간 패턴 분석
        time_columns = ['vst_dt', 'vst_tm', 'otrm_dt', 'otrm_tm']
        if all(col in synth_data.columns for col in time_columns[:2]):
            # 방문 시간 패턴
            synth_data['visit_pattern'] = synth_data['vst_dt'].astype(str) + synth_data['vst_tm'].astype(str)
            
            # 유일한 시간 패턴
            pattern_counts = synth_data['visit_pattern'].value_counts()
            unique_patterns = pattern_counts[pattern_counts == 1]
            results['unique_temporal_patterns'] = len(unique_patterns)
            
            # 반복 방문 패턴 (같은 환자로 추정되는)
            if 'pat_age' in synth_data.columns and 'pat_sex' in synth_data.columns:
                id_cols = ['pat_age', 'pat_sex']
                if 'pat_sarea' in synth_data.columns:
                    id_cols.append('pat_sarea')
                patient_id = synth_data[id_cols].fillna('NA').astype(str).agg('-'.join, axis=1)
                repeated = patient_id.value_counts()
                results['repeated_visits'] = (repeated > 1).sum()
            
            # 위험도 평가
            unique_pct = results['unique_temporal_patterns'] / len(synth_data) * 100
            if unique_pct > 20:
                results['risk_level'] = 'HIGH'
            elif unique_pct > 10:
                results['risk_level'] = 'MEDIUM'
            else:
                results['risk_level'] = 'LOW'
            
            logger.info(f"   - 유일한 시간 패턴: {results['unique_temporal_patterns']} ({unique_pct:.1f}%)")
            logger.info(f"   - 반복 방문 추정: {results['repeated_visits']} 환자")
        
        return results
    
    def _calculate_overall_risk(self):
        """종합 위험도 계산"""
        risk_scores = {
            'LOW': 1,
            'MEDIUM': 2,
            'HIGH': 3,
            'CRITICAL': 4
        }
        
        # 각 카테고리별 점수
        scores = []
        weights = []
        
        # 직접 식별자 (가중치 높음)
        if self.risks['direct_identifiers']:
            scores.append(risk_scores.get(self.risks['direct_identifiers'].get('risk_level', 'MEDIUM'), 2))
            weights.append(3.0)
        
        # 준식별자
        if self.risks['quasi_identifiers']:
            scores.append(risk_scores.get(self.risks['quasi_identifiers'].get('risk_level', 'MEDIUM'), 2))
            weights.append(2.0)
        
        # k-익명성
        if self.risks['k_anonymity']:
            scores.append(risk_scores.get(self.risks['k_anonymity'].get('risk_level', 'MEDIUM'), 2))
            weights.append(2.5)
        
        # l-다양성
        if self.risks['l_diversity']:
            scores.append(risk_scores.get(self.risks['l_diversity'].get('risk_level', 'MEDIUM'), 2))
            weights.append(1.5)
        
        # 연결 공격
        if self.risks['linkage_attacks']:
            scores.append(risk_scores.get(self.risks['linkage_attacks'].get('risk_level', 'MEDIUM'), 2))
            weights.append(2.0)
        
        # 시간 패턴
        if self.risks['temporal_patterns']:
            scores.append(risk_scores.get(self.risks['temporal_patterns'].get('risk_level', 'MEDIUM'), 2))
            weights.append(1.0)
        
        # 가중 평균 계산
        if scores and weights:
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            
            if weighted_score >= 3.5:
                overall_risk = 'CRITICAL'
            elif weighted_score >= 2.5:
                overall_risk = 'HIGH'
            elif weighted_score >= 1.5:
                overall_risk = 'MEDIUM'
            else:
                overall_risk = 'LOW'
        else:
            overall_risk = 'UNKNOWN'
        
        risk_percentage = (weighted_score - 1) / 3 * 100 if scores else 0
        
        return {
            'level': overall_risk,
            'score': weighted_score if scores else 0,
            'percentage': risk_percentage,
            'summary': self._get_risk_summary(overall_risk)
        }
    
    def _get_risk_summary(self, risk_level):
        """위험 수준별 요약"""
        summaries = {
            'CRITICAL': '매우 높은 재식별 위험. 즉시 개선 필요.',
            'HIGH': '높은 재식별 위험. 추가 보호 조치 필요.',
            'MEDIUM': '중간 수준의 재식별 위험. 개선 권장.',
            'LOW': '낮은 재식별 위험. 현재 보호 수준 적절.',
            'UNKNOWN': '위험도 평가 불가. 데이터 확인 필요.'
        }
        return summaries.get(risk_level, '')
    
    def _generate_report(self):
        """평가 보고서 생성"""
        logger.info("\n" + "=" * 80)
        logger.info("재식별 위험 평가 결과")
        logger.info("=" * 80)
        
        overall = self.risks.get('overall_risk', {})
        logger.info(f"\n종합 위험도: {overall.get('level', 'UNKNOWN')} ({overall.get('percentage', 0):.1f}%)")
        logger.info(f"평가: {overall.get('summary', '')}")
        
        logger.info("\n세부 위험 요소:")
        logger.info("-" * 40)
        
        # 각 카테고리별 결과
        categories = [
            ('직접 식별자', 'direct_identifiers'),
            ('준식별자 조합', 'quasi_identifiers'),
            ('k-익명성', 'k_anonymity'),
            ('l-다양성', 'l_diversity'),
            ('연결 공격', 'linkage_attacks'),
            ('시간 패턴', 'temporal_patterns')
        ]
        
        for cat_name, cat_key in categories:
            cat_data = self.risks.get(cat_key, {})
            risk_level = cat_data.get('risk_level', 'UNKNOWN')
            logger.info(f"{cat_name:15} : {risk_level:8}")
        
        # 권장사항
        logger.info("\n권장 개선사항:")
        logger.info("-" * 40)
        recommendations = self._get_recommendations()
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"{i}. {rec}")
        
        # JSON 보고서 저장
        output_dir = Path('outputs/privacy_assessment')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = output_dir / f'privacy_risk_assessment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.risks, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"\n상세 보고서 저장: {report_file}")
    
    def _get_recommendations(self):
        """위험 수준에 따른 권장사항"""
        recommendations = []
        
        # 직접 식별자
        if self.risks['direct_identifiers'].get('found_identifiers'):
            recommendations.append("직접 식별자 완전 제거 또는 암호화 필요")
        
        # k-익명성
        k_anon = self.risks.get('k_anonymity', {})
        if k_anon.get('minimum_k', 5) < 5:
            recommendations.append(f"k-익명성 향상 필요 (현재 최소 k={k_anon.get('minimum_k', 'N/A')})")
        
        # 준식별자
        qi_risks = self.risks.get('quasi_identifiers', {})
        for qi_combo, qi_data in qi_risks.get('unique_combinations', {}).items():
            if qi_data.get('percentage', 0) > 10:
                recommendations.append(f"준식별자 조합 {qi_combo} 일반화 필요")
        
        # 시간 패턴
        temporal = self.risks.get('temporal_patterns', {})
        if temporal.get('unique_temporal_patterns', 0) > 100:
            recommendations.append("시간 정보 일반화 (시간 단위 라운딩) 권장")
        
        # 차분 프라이버시
        recommendations.append("차분 프라이버시 메커니즘 도입 검토")
        
        # 합성 데이터 품질
        recommendations.append("합성 데이터 유용성과 프라이버시 균형 재조정")
        
        return recommendations


def main():
    """재식별 위험 평가 실행"""
    assessor = PrivacyRiskAssessment()
    risks = assessor.assess_all_risks(sample_size=10000)
    
    # 위험 수준에 따른 경고
    overall_risk = risks.get('overall_risk', {}).get('level', 'UNKNOWN')
    if overall_risk in ['CRITICAL', 'HIGH']:
        logger.warning("\n⚠️  경고: 높은 재식별 위험이 감지되었습니다!")
        logger.warning("프로덕션 사용 전 추가 보호 조치가 필요합니다.")
    elif overall_risk == 'MEDIUM':
        logger.info("\n⚠️  주의: 중간 수준의 재식별 위험이 있습니다.")
        logger.info("권장사항을 검토하여 개선을 고려하세요.")
    else:
        logger.info("\n✅ 현재 보호 수준이 적절합니다.")
    
    return risks


if __name__ == "__main__":
    main()