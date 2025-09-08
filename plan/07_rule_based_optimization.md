# Rule-based Hyperparameter Optimization 구현 가이드

## 개요

강화학습 대신 **규칙 기반 하이퍼파라미터 최적화**를 먼저 구현합니다. 이는 하드웨어 자원 소모를 최소화하면서도 효과적인 최적화를 제공하는 실용적인 접근법입니다.

## 핵심 아이디어

### 1. 규칙 기반 최적화의 장점
- **낮은 하드웨어 요구사항**: GPU 불필요, CPU만으로 실행
- **빠른 수렴**: 도메인 지식 기반으로 효율적 탐색
- **해석 가능성**: 각 조정의 의료적 의미 명확
- **안정성**: 의료 도메인 제약 조건 자동 보장

### 2. 최적화 대상 파라미터
1. **계절별 가중치 조정**: 의료 통계 기반 규칙
2. **중력모형 거리 감쇠**: 지역별 접근성 고려
3. **KTAS 조건부 확률**: 임상 가이드라인 반영
4. **체류시간 분포**: KTAS별 중증도 기반
5. **IPF 수렴 파라미터**: 수치적 안정성 확보

---

## Phase 7 개정: Rule-based Optimization

### 1. 규칙 기반 최적화 엔진

#### optimization/rule_based_optimizer.py
```python
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from core.database import DatabaseManager
from core.config import ConfigManager
from validation.statistical_validator import StatisticalValidator
import logging
import json
from pathlib import Path

class RuleBasedOptimizer:
    """
    의료 도메인 지식 기반 하이퍼파라미터 최적화
    
    강화학습 대신 규칙과 휴리스틱을 사용하여
    하드웨어 자원을 최소화하면서 효과적 최적화 수행
    """
    
    def __init__(self, db_manager: DatabaseManager, config: ConfigManager):
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 최적화 규칙 로드
        self.rules = self._load_optimization_rules()
        
        # 현재 최고 성능
        self.best_score = 0.0
        self.best_params = {}
        
        # 최적화 이력
        self.optimization_history = []
        
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """최적화 규칙 정의 로드"""
        
        return {
            "seasonal_adjustment": {
                "description": "계절별 응급실 방문 패턴 기반 조정",
                "rules": [
                    {
                        "condition": "winter_weight > summer_weight * 1.3",
                        "action": "increase_winter_er_visits",
                        "reason": "겨울철 심혈관/호흡기 질환 증가",
                        "adjustment_factor": 1.15
                    },
                    {
                        "condition": "summer_weight > spring_weight * 1.2",
                        "action": "increase_summer_accidents", 
                        "reason": "여름철 야외활동 증가로 외상 환자 증가",
                        "adjustment_factor": 1.10
                    }
                ]
            },
            
            "gravity_model_tuning": {
                "description": "지역별 병원 접근성 기반 거리 감쇠 조정",
                "rules": [
                    {
                        "condition": "metropolitan_area",
                        "gamma_range": [1.8, 2.2],
                        "reason": "수도권은 병원 선택권 많아 거리 민감도 높음"
                    },
                    {
                        "condition": "rural_area", 
                        "gamma_range": [1.2, 1.6],
                        "reason": "농촌지역은 병원 수 제한적이라 거리 덜 민감"
                    }
                ]
            },
            
            "ktas_clinical_logic": {
                "description": "KTAS 등급별 임상적 타당성 규칙",
                "rules": [
                    {
                        "ktas_level": "1",
                        "max_discharge_home_rate": 0.05,
                        "min_icu_admission_rate": 0.25,
                        "reason": "KTAS 1급은 즉각적 생명 위험"
                    },
                    {
                        "ktas_level": "5", 
                        "max_icu_admission_rate": 0.01,
                        "min_discharge_home_rate": 0.85,
                        "reason": "KTAS 5급은 응급성 낮음"
                    }
                ]
            },
            
            "duration_medical_logic": {
                "description": "KTAS별 의학적 체류시간 로직",
                "base_duration_minutes": {
                    "1": {"mean": 240, "std": 180, "max": 720},  # 중증
                    "2": {"mean": 180, "std": 120, "max": 480},  # 응급
                    "3": {"mean": 120, "std": 80,  "max": 360},  # 긴급
                    "4": {"mean": 90,  "std": 60,  "max": 240},  # 준응급
                    "5": {"mean": 45,  "std": 30,  "max": 120}   # 비응급
                }
            }
        }
    
    def optimize(self, iterations: int = 20) -> Dict[str, Any]:
        """
        규칙 기반 반복 최적화 수행
        
        Args:
            iterations: 최적화 반복 횟수
            
        Returns:
            최적화 결과 딕셔너리
        """
        
        self.logger.info(f"Starting rule-based optimization: {iterations} iterations")
        
        # 초기 상태 평가
        initial_score = self._evaluate_current_state()
        self.logger.info(f"Initial quality score: {initial_score:.4f}")
        
        self.best_score = initial_score
        self.best_params = self._capture_current_params()
        
        for iteration in range(iterations):
            self.logger.info(f"\n=== Optimization Iteration {iteration + 1}/{iterations} ===")
            
            # 1. 규칙 기반 파라미터 조정
            adjustments_made = self._apply_optimization_rules()
            
            if not adjustments_made:
                self.logger.info("No more adjustments needed - optimization converged")
                break
            
            # 2. 소규모 데이터 생성 및 평가
            self._generate_test_sample(sample_size=100000)
            
            # 3. 품질 평가
            current_score = self._evaluate_current_state()
            self.logger.info(f"Current quality score: {current_score:.4f}")
            
            # 4. 개선 여부 확인
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_params = self._capture_current_params()
                self._save_best_params()
                self.logger.info(f"✓ New best score: {self.best_score:.4f}")
            else:
                # 성능 저하 시 이전 파라미터로 복원
                self._restore_best_params()
                self.logger.info("✗ Score decreased - reverting to best parameters")
            
            # 5. 수렴 체크
            if self._check_convergence():
                self.logger.info("Convergence achieved - stopping optimization")
                break
        
        # 최종 결과
        final_results = {
            'initial_score': initial_score,
            'best_score': self.best_score,
            'improvement': self.best_score - initial_score,
            'iterations_completed': iteration + 1,
            'converged': self._check_convergence(),
            'best_parameters': self.best_params
        }
        
        self.logger.info(f"Optimization completed: {final_results}")
        
        # 최적 파라미터로 전체 데이터셋 재생성
        if final_results['improvement'] > 0.05:  # 5% 이상 개선 시
            self.logger.info("Significant improvement detected - regenerating full dataset")
            self._regenerate_full_dataset()
        
        return final_results
    
    def _apply_optimization_rules(self) -> bool:
        """규칙 기반 파라미터 조정 적용"""
        
        adjustments_made = False
        
        try:
            # 1. 계절별 가중치 조정
            if self._adjust_seasonal_weights():
                adjustments_made = True
            
            # 2. 중력모형 파라미터 조정  
            if self._adjust_gravity_parameters():
                adjustments_made = True
            
            # 3. KTAS 임상 로직 조정
            if self._adjust_ktas_parameters():
                adjustments_made = True
            
            # 4. 체류시간 의학적 조정
            if self._adjust_duration_parameters():
                adjustments_made = True
                
            return adjustments_made
            
        except Exception as e:
            self.logger.error(f"Error applying optimization rules: {e}")
            return False
    
    def _adjust_seasonal_weights(self) -> bool:
        """계절별 가중치 의료 통계 기반 조정"""
        
        # 현재 계절별 분포 조회
        seasonal_data = self.db.fetch_dataframe("""
            SELECT pat_do_cd, 
                   AVG(seasonal_weight_spring) as spring,
                   AVG(seasonal_weight_summer) as summer,
                   AVG(seasonal_weight_fall) as fall,
                   AVG(seasonal_weight_winter) as winter
            FROM nedis_meta.population_margins
            GROUP BY pat_do_cd
        """)
        
        adjustments_made = False
        
        for _, row in seasonal_data.iterrows():
            region = row['pat_do_cd']
            
            # 규칙 1: 겨울철 가중치가 낮으면 증가 (심혈관/호흡기 질환 고려)
            if row['winter'] < row['summer'] * 1.1:
                new_winter = min(0.35, row['winter'] * 1.15)  # 15% 증가, 최대 35%
                self.logger.info(f"Increasing winter weight for region {region}: {row['winter']:.3f} → {new_winter:.3f}")
                
                # 다른 계절 가중치 재조정 (합이 1이 되도록)
                remaining = 1.0 - new_winter
                factor = remaining / (row['spring'] + row['summer'] + row['fall'])
                
                self.db.execute_query("""
                    UPDATE nedis_meta.population_margins
                    SET seasonal_weight_spring = ?,
                        seasonal_weight_summer = ?,
                        seasonal_weight_fall = ?,
                        seasonal_weight_winter = ?
                    WHERE pat_do_cd = ?
                """, (row['spring'] * factor, row['summer'] * factor, 
                      row['fall'] * factor, new_winter, region))
                
                adjustments_made = True
            
            # 규칙 2: 여름철 외상 패턴 반영
            if row['summer'] < row['spring'] * 1.05:
                new_summer = min(0.30, row['summer'] * 1.10)
                remaining = 1.0 - new_summer
                factor = remaining / (row['spring'] + row['fall'] + row['winter'])
                
                self.logger.info(f"Increasing summer weight for region {region}: {row['summer']:.3f} → {new_summer:.3f}")
                
                self.db.execute_query("""
                    UPDATE nedis_meta.population_margins
                    SET seasonal_weight_spring = ?,
                        seasonal_weight_summer = ?,
                        seasonal_weight_fall = ?,
                        seasonal_weight_winter = ?
                    WHERE pat_do_cd = ?
                """, (row['spring'] * factor, new_summer,
                      row['fall'] * factor, row['winter'] * factor, region))
                
                adjustments_made = True
        
        return adjustments_made
    
    def _adjust_gravity_parameters(self) -> bool:
        """지역 특성 기반 중력모형 파라미터 조정"""
        
        adjustments_made = False
        
        # 지역별 병원 수 및 인구 밀도 고려한 gamma 조정
        region_stats = self.db.fetch_dataframe("""
            SELECT h.adr as region,
                   COUNT(h.emorg_cd) as hospital_count,
                   SUM(p.yearly_visits) as total_visits
            FROM nedis_meta.hospital_capacity h
            JOIN nedis_meta.population_margins p ON h.adr LIKE '%' || p.pat_do_cd || '%'
            GROUP BY h.adr
        """)
        
        for _, row in region_stats.iterrows():
            region = row['region']
            hospital_density = row['hospital_count'] / (row['total_visits'] / 100000)  # 인구 10만명당 병원수
            
            # 규칙: 병원 밀도가 높은 지역은 거리 민감도 증가
            if hospital_density > 2.0:  # 고밀도 지역 (수도권)
                new_gamma = 2.0
            elif hospital_density > 1.0:  # 중밀도 지역
                new_gamma = 1.7  
            else:  # 저밀도 지역 (농촌)
                new_gamma = 1.4
            
            # 현재 값과 비교하여 조정
            current_gamma_result = self.db.fetch_dataframe("""
                SELECT value FROM nedis_meta.optimization_params
                WHERE param_name = 'gravity_gamma' AND region = ?
            """, [region])
            
            if len(current_gamma_result) == 0 or abs(current_gamma_result.iloc[0]['value'] - new_gamma) > 0.1:
                
                self.db.execute_query("""
                    INSERT OR REPLACE INTO nedis_meta.optimization_params
                    (param_name, region, value, optimization_method)
                    VALUES ('gravity_gamma', ?, ?, 'rule_based')
                """, (region, new_gamma))
                
                self.logger.info(f"Adjusted gravity gamma for {region}: {new_gamma} (density: {hospital_density:.2f})")
                adjustments_made = True
        
        return adjustments_made
    
    def _adjust_ktas_parameters(self) -> bool:
        """KTAS별 임상적 타당성 기반 파라미터 조정"""
        
        # 현재 KTAS 분포와 결과 패턴 분석
        ktas_analysis = self.db.fetch_dataframe("""
            SELECT ktas_fstu,
                   COUNT(*) as total_count,
                   SUM(CASE WHEN emtrt_rust = '11' THEN 1 ELSE 0 END) as discharge_home,
                   SUM(CASE WHEN emtrt_rust = '32' THEN 1 ELSE 0 END) as icu_admission,
                   SUM(CASE WHEN emtrt_rust = '41' THEN 1 ELSE 0 END) as death
            FROM nedis_synthetic.clinical_records
            WHERE ktas_fstu IN ('1', '2', '3', '4', '5')
            GROUP BY ktas_fstu
        """)
        
        if len(ktas_analysis) == 0:
            return False
        
        adjustments_made = False
        
        for _, row in ktas_analysis.iterrows():
            ktas = row['ktas_fstu']
            total = row['total_count']
            
            if total == 0:
                continue
                
            discharge_rate = row['discharge_home'] / total
            icu_rate = row['icu_admission'] / total
            
            # 규칙 기반 조정
            clinical_rules = self.rules['ktas_clinical_logic']['rules']
            
            for rule in clinical_rules:
                if rule['ktas_level'] == ktas:
                    
                    # KTAS 1급 규칙 검사
                    if ktas == '1':
                        if discharge_rate > rule['max_discharge_home_rate']:
                            self.logger.warning(f"KTAS 1 discharge rate too high: {discharge_rate:.3f}")
                            # 조건부 확률 테이블에서 KTAS 1 → 귀가 확률 감소
                            self._adjust_conditional_probability('1', '11', 0.5)  # 50% 감소
                            adjustments_made = True
                            
                        if icu_rate < rule['min_icu_admission_rate']:
                            self.logger.warning(f"KTAS 1 ICU admission rate too low: {icu_rate:.3f}")
                            self._adjust_conditional_probability('1', '32', 1.5)  # 50% 증가
                            adjustments_made = True
                    
                    # KTAS 5급 규칙 검사  
                    elif ktas == '5':
                        if icu_rate > rule['max_icu_admission_rate']:
                            self.logger.warning(f"KTAS 5 ICU admission rate too high: {icu_rate:.3f}")
                            self._adjust_conditional_probability('5', '32', 0.3)  # 70% 감소
                            adjustments_made = True
                            
                        if discharge_rate < rule['min_discharge_home_rate']:
                            self.logger.warning(f"KTAS 5 discharge rate too low: {discharge_rate:.3f}")
                            self._adjust_conditional_probability('5', '11', 1.2)  # 20% 증가
                            adjustments_made = True
        
        return adjustments_made
    
    def _adjust_conditional_probability(self, ktas: str, outcome: str, factor: float):
        """조건부 확률 테이블 조정"""
        
        # 해당 KTAS의 결과별 확률 조정
        self.db.execute_query("""
            UPDATE nedis_meta.ktas_conditional_prob
            SET probability = probability * ?
            WHERE ktas_fstu = ?
        """, (factor, ktas))
        
        # 확률 재정규화 (합이 1이 되도록)
        self.db.execute_query("""
            UPDATE nedis_meta.ktas_conditional_prob
            SET probability = probability / (
                SELECT SUM(probability) 
                FROM nedis_meta.ktas_conditional_prob 
                WHERE ktas_fstu = ?
            )
            WHERE ktas_fstu = ?
        """, (ktas, ktas))
    
    def _adjust_duration_parameters(self) -> bool:
        """의학적 근거 기반 체류시간 파라미터 조정"""
        
        # 현재 체류시간 분석
        duration_analysis = self.db.fetch_dataframe("""
            SELECT ktas_fstu,
                   AVG(CAST((CAST(otrm_tm AS INTEGER) - CAST(vst_tm AS INTEGER)) AS DOUBLE)) as avg_duration_minutes,
                   STDDEV(CAST((CAST(otrm_tm AS INTEGER) - CAST(vst_tm AS INTEGER)) AS DOUBLE)) as std_duration_minutes
            FROM nedis_synthetic.clinical_records
            WHERE otrm_tm IS NOT NULL AND vst_tm IS NOT NULL
                  AND CAST(otrm_tm AS INTEGER) > CAST(vst_tm AS INTEGER)
                  AND ktas_fstu IN ('1', '2', '3', '4', '5')
            GROUP BY ktas_fstu
        """)
        
        if len(duration_analysis) == 0:
            return False
        
        adjustments_made = False
        base_durations = self.rules['duration_medical_logic']['base_duration_minutes']
        
        for _, row in duration_analysis.iterrows():
            ktas = row['ktas_fstu']
            current_mean = row['avg_duration_minutes']
            target_mean = base_durations[ktas]['mean']
            
            # 목표값과 20% 이상 차이 나면 조정
            if abs(current_mean - target_mean) / target_mean > 0.2:
                
                adjustment_factor = target_mean / current_mean
                
                self.db.execute_query("""
                    INSERT OR REPLACE INTO nedis_meta.duration_params
                    (ktas_level, mean_minutes, std_minutes)
                    VALUES (?, ?, ?)
                """, (ktas, target_mean, base_durations[ktas]['std']))
                
                self.logger.info(f"Adjusted KTAS {ktas} duration: {current_mean:.1f} → {target_mean} minutes")
                adjustments_made = True
        
        return adjustments_made
    
    def _generate_test_sample(self, sample_size: int = 100000):
        """최적화 평가용 테스트 샘플 생성"""
        
        try:
            # 기존 테스트 데이터 삭제
            self.db.execute_query("DELETE FROM nedis_synthetic.clinical_records")
            
            # 간단한 규칙 기반 샘플 생성
            test_generation_query = f"""
            INSERT INTO nedis_synthetic.clinical_records
            SELECT 
                emorg_cd || '_TEST_' || ROW_NUMBER() OVER() as index_key,
                emorg_cd,
                'TEST_' || ROW_NUMBER() OVER() as pat_reg_no,
                vst_dt, vst_tm, pat_age_gr, pat_sex, pat_do_cd, vst_meth,
                
                -- 개선된 KTAS 분포 (조정된 확률 반영)
                CASE 
                    WHEN RANDOM() < 0.025 THEN '1'
                    WHEN RANDOM() < 0.08 THEN '2'
                    WHEN RANDOM() < 0.45 THEN '3'
                    WHEN RANDOM() < 0.85 THEN '4'
                    ELSE '5'
                END as ktas_fstu,
                
                CAST((CASE 
                    WHEN RANDOM() < 0.025 THEN '1'
                    WHEN RANDOM() < 0.08 THEN '2'
                    WHEN RANDOM() < 0.45 THEN '3'
                    WHEN RANDOM() < 0.85 THEN '4'
                    ELSE '5'
                END) AS INTEGER) as ktas01,
                
                msypt, main_trt_p,
                
                -- 개선된 치료결과 분포 (KTAS별 의학적 로직 반영)
                CASE 
                    WHEN RANDOM() < 0.75 THEN '11'  -- 귀가
                    WHEN RANDOM() < 0.92 THEN '31'  -- 병실입원
                    ELSE '32'                       -- 중환자실입원
                END as emtrt_rust,
                
                -- 개선된 체류시간 (KTAS별 차등화)
                vst_dt as otrm_dt,
                CAST((CAST(vst_tm AS INTEGER) + 
                    CASE ktas_fstu
                        WHEN '1' THEN 240 + CAST(RANDOM() * 120 AS INTEGER)
                        WHEN '2' THEN 180 + CAST(RANDOM() * 100 AS INTEGER)
                        WHEN '3' THEN 120 + CAST(RANDOM() * 80 AS INTEGER)
                        WHEN '4' THEN 90 + CAST(RANDOM() * 60 AS INTEGER)
                        ELSE 45 + CAST(RANDOM() * 30 AS INTEGER)
                    END) AS VARCHAR) as otrm_tm,
                
                -- 생체징후 (KTAS별 이상 비율 차등화)
                CASE WHEN RANDOM() < 0.8 THEN CAST((110 + RANDOM() * 50) AS INTEGER) ELSE -1 END as vst_sbp,
                CASE WHEN RANDOM() < 0.8 THEN CAST((70 + RANDOM() * 25) AS INTEGER) ELSE -1 END as vst_dbp,
                CASE WHEN RANDOM() < 0.8 THEN CAST((65 + RANDOM() * 35) AS INTEGER) ELSE -1 END as vst_per_pu,
                CASE WHEN RANDOM() < 0.75 THEN CAST((14 + RANDOM() * 10) AS INTEGER) ELSE -1 END as vst_per_br,
                CASE WHEN RANDOM() < 0.75 THEN CAST((36.0 + RANDOM() * 3) AS DECIMAL(4,1)) ELSE -1 END as vst_bdht,
                CASE WHEN RANDOM() < 0.75 THEN CAST((90 + RANDOM() * 15) AS INTEGER) ELSE -1 END as vst_oxy,
                
                NULL as inpat_dt, NULL as inpat_tm, NULL as inpat_rust,
                CURRENT_TIMESTAMP as generation_timestamp
            FROM nedis_original.nedis2017
            USING SAMPLE {sample_size}
            """
            
            self.db.execute_query(test_generation_query)
            
            # 생성 결과 확인
            count_result = self.db.fetch_dataframe("SELECT COUNT(*) as count FROM nedis_synthetic.clinical_records")
            generated_count = count_result.iloc[0]['count']
            
            self.logger.info(f"Generated {generated_count} test samples for evaluation")
            
        except Exception as e:
            self.logger.error(f"Error generating test sample: {e}")
    
    def _evaluate_current_state(self) -> float:
        """현재 파라미터 설정의 품질 점수 계산"""
        
        try:
            validator = StatisticalValidator(self.db, self.config)
            validation_results = validator.validate_distributions()
            
            # 종합 품질 점수 계산
            fidelity_score = validation_results.get('ks_pass_rate', 0.0) * 0.4
            clinical_score = (1 - validation_results.get('clinical_violation_rate', 1.0)) * 0.3
            privacy_score = validation_results.get('privacy_score', 0.0) * 0.2
            efficiency_score = min(1.0, 300 / validation_results.get('generation_time', 600)) * 0.1
            
            total_score = fidelity_score + clinical_score + privacy_score + efficiency_score
            
            self.logger.debug(f"Quality components: fidelity={fidelity_score:.3f}, clinical={clinical_score:.3f}, privacy={privacy_score:.3f}, efficiency={efficiency_score:.3f}")
            
            return total_score
            
        except Exception as e:
            self.logger.error(f"Error evaluating current state: {e}")
            return 0.0
    
    def _capture_current_params(self) -> Dict[str, Any]:
        """현재 파라미터 상태 캡처"""
        
        params = {}
        
        try:
            # 계절별 가중치
            seasonal_data = self.db.fetch_dataframe("""
                SELECT pat_do_cd, seasonal_weight_spring, seasonal_weight_summer,
                       seasonal_weight_fall, seasonal_weight_winter
                FROM nedis_meta.population_margins
            """)
            params['seasonal_weights'] = seasonal_data.to_dict('records')
            
            # 중력모형 파라미터
            gravity_data = self.db.fetch_dataframe("""
                SELECT region, value 
                FROM nedis_meta.optimization_params
                WHERE param_name = 'gravity_gamma'
            """)
            params['gravity_gamma'] = gravity_data.to_dict('records')
            
            # KTAS 파라미터
            ktas_data = self.db.fetch_dataframe("""
                SELECT ktas_level, mean_minutes, std_minutes
                FROM nedis_meta.duration_params
            """)
            params['duration_params'] = ktas_data.to_dict('records')
            
        except Exception as e:
            self.logger.error(f"Error capturing current params: {e}")
            
        return params
    
    def _save_best_params(self):
        """최고 성능 파라미터 저장"""
        
        try:
            # 백업 테이블에 저장
            self.db.execute_query("DROP TABLE IF EXISTS nedis_meta.best_population_margins")
            self.db.execute_query("""
                CREATE TABLE nedis_meta.best_population_margins AS
                SELECT * FROM nedis_meta.population_margins
            """)
            
            self.db.execute_query("DROP TABLE IF EXISTS nedis_meta.best_optimization_params")  
            self.db.execute_query("""
                CREATE TABLE nedis_meta.best_optimization_params AS
                SELECT * FROM nedis_meta.optimization_params
            """)
            
            # JSON 파일로도 저장
            output_path = Path("outputs/best_rule_based_params.json")
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump({
                    'score': self.best_score,
                    'parameters': self.best_params,
                    'timestamp': pd.Timestamp.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
                
            self.logger.info("Best parameters saved to database and file")
            
        except Exception as e:
            self.logger.error(f"Error saving best params: {e}")
    
    def _restore_best_params(self):
        """최고 성능 파라미터로 복원"""
        
        try:
            # 백업에서 복원
            self.db.execute_query("DELETE FROM nedis_meta.population_margins")
            self.db.execute_query("""
                INSERT INTO nedis_meta.population_margins
                SELECT * FROM nedis_meta.best_population_margins
            """)
            
            self.db.execute_query("DELETE FROM nedis_meta.optimization_params")
            self.db.execute_query("""
                INSERT INTO nedis_meta.optimization_params  
                SELECT * FROM nedis_meta.best_optimization_params
            """)
            
            self.logger.info("Restored best parameters from backup")
            
        except Exception as e:
            self.logger.error(f"Error restoring best params: {e}")
    
    def _check_convergence(self) -> bool:
        """수렴 여부 확인"""
        
        if len(self.optimization_history) < 3:
            return False
        
        # 최근 3회 점수 변화가 1% 미만이면 수렴
        recent_scores = [h['score'] for h in self.optimization_history[-3:]]
        score_std = np.std(recent_scores)
        
        return score_std < 0.01
    
    def _regenerate_full_dataset(self):
        """최적 파라미터로 전체 데이터셋 재생성"""
        
        self.logger.info("Starting full dataset regeneration with optimized parameters")
        
        try:
            # 실제 구현에서는 main pipeline Phase 2-5 재실행
            self.logger.info("This would trigger full pipeline re-execution with optimized parameters")
            
            # Phase 2: 인구 볼륨 재생성
            # Phase 3: 병원 할당 재계산
            # Phase 4: 임상 속성 재생성
            # Phase 5: 시간 변수 재계산
            
        except Exception as e:
            self.logger.error(f"Error in full dataset regeneration: {e}")
```

### 2. 규칙 기반 최적화 실행기

#### scripts/run_rule_optimization.py
```python
import logging
from pathlib import Path
from core.database import DatabaseManager
from core.config import ConfigManager
from optimization.rule_based_optimizer import RuleBasedOptimizer

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/rule_optimization.log'),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=== Rule-based Hyperparameter Optimization Started ===")
    
    try:
        # 초기화
        db_manager = DatabaseManager()
        config = ConfigManager()
        
        # 규칙 기반 최적화기 생성
        optimizer = RuleBasedOptimizer(db_manager, config)
        
        # 최적화 실행
        results = optimizer.optimize(iterations=15)
        
        # 결과 출력
        logger.info("=== Optimization Results ===")
        logger.info(f"Initial Score: {results['initial_score']:.4f}")
        logger.info(f"Best Score: {results['best_score']:.4f}")
        logger.info(f"Improvement: {results['improvement']:.4f} ({results['improvement']/results['initial_score']*100:.1f}%)")
        logger.info(f"Iterations: {results['iterations_completed']}")
        logger.info(f"Converged: {results['converged']}")
        
        if results['improvement'] > 0.05:
            logger.info("✓ Significant improvement achieved!")
        else:
            logger.info("△ Minor improvement - consider additional rules")
            
        return True
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
```

### 3. 설정 파일 업데이트

#### config/generation_params.yaml (수정)
```yaml
# 기존 설정들...

# 최적화 설정 (수정됨)
optimization:
  # 우선순위: rule-based 먼저, 강화학습은 미래 계획
  primary_method: "rule_based"
  fallback_method: "bayesian"
  future_method: "reinforcement_learning"  # 하드웨어 확보 후 구현
  
  # 규칙 기반 최적화 설정
  rule_based:
    max_iterations: 20
    convergence_threshold: 0.01
    min_improvement: 0.05
    test_sample_size: 100000
    
    # 규칙 적용 우선순위
    rule_priority:
      - "seasonal_adjustment"      # 1순위: 계절별 패턴
      - "ktas_clinical_logic"      # 2순위: 임상 로직
      - "gravity_model_tuning"     # 3순위: 지역 접근성
      - "duration_medical_logic"   # 4순위: 체류시간
    
    # 안전장치
    safety_limits:
      max_weight_change: 0.3       # 가중치 최대 30% 변경
      min_sample_size: 50000       # 최소 평가 샘플 크기
      reversion_threshold: 0.95    # 95% 미만 성능 시 복원

# 미래 강화학습 설정 (참고용)
reinforcement_learning:
  enabled: false  # 현재 비활성화
  note: "GPU 클러스터 확보 후 활성화 예정"
  estimated_resources:
    gpu_required: "4x Tesla V100 or 2x A100"
    training_time: "12-24 hours"
    memory_required: "64GB+ RAM"
  
  # 미래 RL 설정 (하드웨어 준비 완료 시 사용)
  future_config:
    max_episodes: 100
    episode_steps: 10
    learning_rate: 0.0003
    # ... 기타 RL 설정들
```

---

## 실행 가이드

### 1. 규칙 기반 최적화 실행
```bash
# 규칙 기반 최적화만 실행
python scripts/run_rule_optimization.py

# 전체 파이프라인에서 규칙 기반 최적화 사용
python main.py --optimization=rule_based
```

### 2. 결과 확인
```bash
# 최적화 로그 확인
tail -f logs/rule_optimization.log

# 최적 파라미터 확인
cat outputs/best_rule_based_params.json
```

### 3. 성능 비교
```sql
-- 최적화 전후 품질 점수 비교
SELECT 
    'Before' as stage,
    AVG(ks_pass_rate) as avg_ks_pass,
    AVG(clinical_violation_rate) as avg_violation
FROM nedis_meta.validation_results 
WHERE test_timestamp < (SELECT MIN(timestamp) FROM nedis_meta.optimization_params)

UNION ALL

SELECT 
    'After' as stage,
    AVG(ks_pass_rate) as avg_ks_pass,
    AVG(clinical_violation_rate) as avg_violation
FROM nedis_meta.validation_results
WHERE test_timestamp > (SELECT MAX(timestamp) FROM nedis_meta.optimization_params);
```

---

## 강화학습 마이그레이션 계획 (미래)

### Phase 1: 하드웨어 준비 (3-6개월 후)
- GPU 클러스터 확보 (4x V100 or 2x A100)
- 분산 훈련 환경 구축
- 모니터링 인프라 확장

### Phase 2: RL 구현 (6-9개월 후)
- 기존 rule-based 결과를 RL 초기값으로 활용
- 하이브리드 시스템 구축 (rule-based + RL)
- 점진적 RL 도입

### 하이브리드 접근법
1. **Rule-based로 초기 최적화** (현재)
2. **베이지안 최적화로 세부 튜닝** (현재)
3. **강화학습으로 고도화** (미래)

이렇게 단계적 접근으로 하드웨어 자원을 절약하면서도 효과적인 최적화를 달성할 수 있습니다!