# NEDIS Synthetic Data Generation 개발 계획서

## 프로젝트 구조 및 환경 설정

### 디렉토리 구조 생성
```bash
nedis-synthetic/
├── config/
│   ├── generation_params.yaml
│   ├── clinical_rules.json
│   └── hospital_metadata.json
├── src/
│   ├── population/
│   ├── temporal/
│   ├── allocation/
│   ├── clinical/
│   ├── validation/
│   └── optimization/
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
└── tests/
```

### DuckDB 스키마 정의
```sql
-- 원본 데이터 스키마 (읽기 전용)
CREATE SCHEMA IF NOT EXISTS nedis_original;

-- 합성 데이터 스키마
CREATE SCHEMA IF NOT EXISTS nedis_synthetic;

-- 메타데이터 스키마
CREATE SCHEMA IF NOT EXISTS nedis_meta;

-- nedis_meta.hospital_capacity 테이블 생성
CREATE TABLE nedis_meta.hospital_capacity (
    emorg_cd VARCHAR PRIMARY KEY,
    hospname VARCHAR NOT NULL,
    gubun VARCHAR NOT NULL,  -- '권역센터', '지역센터', '지역기관'
    adr VARCHAR NOT NULL,     -- 시도 (서울, 경기 등)
    daily_capacity_mean INTEGER,
    daily_capacity_std INTEGER,
    ktas1_capacity INTEGER,
    ktas2_capacity INTEGER,
    attractiveness_score DOUBLE  -- 중력모형용
);

-- nedis_meta.population_margins 테이블 생성
CREATE TABLE nedis_meta.population_margins (
    pat_do_cd VARCHAR,        -- 시도 코드 (41, 11, 28 등)
    pat_age_gr VARCHAR,       -- 연령군 (01, 09, 10, 20, 30, 40, 50, 60, 70, 80, 90)
    pat_sex VARCHAR,          -- 성별 (M, F)
    yearly_visits INTEGER,
    seasonal_weight_spring DOUBLE,
    seasonal_weight_summer DOUBLE,
    seasonal_weight_fall DOUBLE,
    seasonal_weight_winter DOUBLE,
    weekday_weight DOUBLE,
    weekend_weight DOUBLE,
    PRIMARY KEY (pat_do_cd, pat_age_gr, pat_sex)
);
```

---

## Phase 1: 데이터 프로파일링 및 메타데이터 추출

### Task 1.1: 원본 데이터 통계 추출
```python
# src/population/profiler.py 생성

class NEDISProfiler:
    def __init__(self, conn):
        """
        conn: DuckDB connection 객체
        """
        self.conn = conn
    
    def extract_population_margins(self):
        """
        nedis_original.nedis2017 테이블에서 다음 집계 실행:
        
        SELECT 
            pat_do_cd,
            pat_age_gr,
            pat_sex,
            COUNT(*) as yearly_visits,
            -- 계절별 가중치 계산
            SUM(CASE WHEN MONTH(vst_dt::DATE) IN (3,4,5) THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) as seasonal_weight_spring,
            SUM(CASE WHEN MONTH(vst_dt::DATE) IN (6,7,8) THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) as seasonal_weight_summer,
            SUM(CASE WHEN MONTH(vst_dt::DATE) IN (9,10,11) THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) as seasonal_weight_fall,
            SUM(CASE WHEN MONTH(vst_dt::DATE) IN (12,1,2) THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) as seasonal_weight_winter,
            -- 요일별 가중치 계산
            SUM(CASE WHEN DAYOFWEEK(vst_dt::DATE) IN (1,7) THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) as weekend_weight
        FROM nedis_original.nedis2017
        WHERE pat_do_cd != '' AND pat_age_gr != '' AND pat_sex IN ('M', 'F')
        GROUP BY pat_do_cd, pat_age_gr, pat_sex
        
        결과를 nedis_meta.population_margins에 INSERT
        """
        pass
    
    def extract_hospital_statistics(self):
        """
        각 emorg_cd별 일평균, 표준편차, KTAS별 capacity 계산:
        
        WITH daily_counts AS (
            SELECT 
                emorg_cd,
                vst_dt,
                COUNT(*) as daily_visits,
                SUM(CASE WHEN ktas_fstu = '1' THEN 1 ELSE 0 END) as ktas1_count,
                SUM(CASE WHEN ktas_fstu = '2' THEN 1 ELSE 0 END) as ktas2_count
            FROM nedis_original.nedis2017
            GROUP BY emorg_cd, vst_dt
        )
        SELECT 
            emorg_cd,
            AVG(daily_visits) as daily_capacity_mean,
            STDDEV(daily_visits) as daily_capacity_std,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ktas1_count) as ktas1_capacity,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ktas2_count) as ktas2_capacity
        FROM daily_counts
        GROUP BY emorg_cd
        
        결과를 nedis_meta.hospital_capacity에 UPDATE
        """
        pass
```

### Task 1.2: 조건부 확률 테이블 생성
```python
# src/clinical/conditional_probability.py 생성

class ConditionalProbabilityExtractor:
    def __init__(self, conn):
        self.conn = conn
    
    def create_ktas_probability_table(self):
        """
        nedis_meta.ktas_conditional_prob 테이블 생성:
        
        CREATE TABLE nedis_meta.ktas_conditional_prob (
            pat_age_gr VARCHAR,
            pat_sex VARCHAR,
            gubun VARCHAR,           -- 병원 종별
            vst_meth VARCHAR,        -- 내원수단 (1:119구급차, 3:기타구급차, 6:도보 등)
            ktas_fstu VARCHAR,       -- KTAS 등급 (1,2,3,4,5)
            probability DOUBLE,
            sample_count INTEGER,
            PRIMARY KEY (pat_age_gr, pat_sex, gubun, vst_meth, ktas_fstu)
        );
        
        베이지안 평활 적용 (α = 1.0):
        probability = (count + α) / (total + α * n_categories)
        """
        pass
    
    def create_diagnosis_probability_table(self):
        """
        nedis_meta.diagnosis_conditional_prob 테이블 생성:
        
        CREATE TABLE nedis_meta.diagnosis_conditional_prob (
            pat_age_gr VARCHAR,
            pat_sex VARCHAR,
            gubun VARCHAR,
            ktas_fstu VARCHAR,
            diagnosis_code VARCHAR,  -- diag_er 테이블의 diagnosis_code
            probability DOUBLE,
            is_primary BOOLEAN,      -- position = 1인 경우
            sample_count INTEGER
        );
        
        diag_er 테이블과 JOIN하여 주진단(position=1) 빈도 계산
        희귀 진단(count < 10)은 상위 3자리로 그룹화
        """
        pass
```

---

## Phase 2: 인구 및 시간 패턴 생성

### Task 2.1: Dirichlet-Multinomial 인구 생성기
```python
# src/population/generator.py 생성

import numpy as np
from scipy.stats import dirichlet

class PopulationVolumeGenerator:
    def __init__(self, conn, alpha_smoothing=1.0):
        self.conn = conn
        self.alpha = alpha_smoothing
        
    def generate_yearly_volumes(self, target_year_total=9_200_000):
        """
        1. nedis_meta.population_margins에서 pat_do_cd별 비율 로드
        
        2. 각 시도별 Dirichlet 샘플링:
           query = '''
           SELECT pat_age_gr, pat_sex, yearly_visits 
           FROM nedis_meta.population_margins 
           WHERE pat_do_cd = ?
           '''
           
        3. Dirichlet 파라미터 설정:
           alpha_vector = observed_counts + self.alpha
           pi_samples = dirichlet.rvs(alpha_vector)
           
        4. Multinomial 샘플링:
           n_ras = np.random.multinomial(region_total, pi_samples)
           
        5. 결과를 nedis_synthetic.yearly_volumes 테이블에 저장:
           CREATE TABLE nedis_synthetic.yearly_volumes (
               pat_do_cd VARCHAR,
               pat_age_gr VARCHAR,
               pat_sex VARCHAR,
               synthetic_yearly_count INTEGER,
               generation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
           );
        """
        pass
```

### Task 2.2: 비균질 포아송 과정(NHPP) 일별 분해
```python
# src/temporal/nhpp_generator.py 생성

class NHPPTemporalGenerator:
    def __init__(self, conn):
        self.conn = conn
        
    def generate_daily_events(self, year=2017):
        """
        1. nedis_synthetic.yearly_volumes와 nedis_meta.population_margins JOIN
        
        2. 각 (pat_do_cd, pat_age_gr, pat_sex)별로:
           - 기본 강도: λ_base = yearly_count / 365
           - 계절 가중치: S_m = seasonal_weight_{season}
           - 요일 가중치: W_d = weekday_weight or (1 - weekday_weight)
           - 공휴일 가중치: H_u = 1.2 (공휴일) or 1.0
           
        3. 일별 강도 계산:
           λ(t) = λ_base * S_m(t) * W_d(t) * H_u(t)
           
        4. 포아송 샘플링:
           daily_counts = np.random.poisson(λ(t))
           
        5. Rescaling (연간 총합 맞추기):
           scaling_factor = yearly_count / sum(daily_counts)
           adjusted_counts = daily_counts * scaling_factor
           
        6. 결과 저장:
           CREATE TABLE nedis_synthetic.daily_volumes (
               vst_dt VARCHAR,  -- YYYYMMDD 형식
               pat_do_cd VARCHAR,
               pat_age_gr VARCHAR,
               pat_sex VARCHAR,
               synthetic_daily_count INTEGER,
               lambda_value DOUBLE
           );
        """
        pass
```

---

## Phase 3: 병원 할당 및 용량 제약

### Task 3.1: 중력모형(Huff Model) 구현
```python
# src/allocation/gravity_model.py 생성

class HospitalGravityAllocator:
    def __init__(self, conn, gamma=1.5):
        """
        gamma: 거리 감쇠 파라미터 (1.0 ~ 2.0 권장)
        """
        self.conn = conn
        self.gamma = gamma
        
    def calculate_allocation_probabilities(self):
        """
        1. 거리 매트릭스 생성 (시도 중심점 기준):
           CREATE TABLE nedis_meta.distance_matrix (
               from_do_cd VARCHAR,
               to_emorg_cd VARCHAR,
               distance_km DOUBLE,
               travel_time_min DOUBLE
           );
           
        2. 병원 매력도 계산:
           query = '''
           SELECT 
               emorg_cd,
               gubun,
               daily_capacity_mean * 
               CASE gubun 
                   WHEN '권역센터' THEN 2.0
                   WHEN '지역센터' THEN 1.5
                   WHEN '지역기관' THEN 1.0
               END as attractiveness
           FROM nedis_meta.hospital_capacity
           '''
           
        3. Huff 확률 계산:
           P(h|r,a,s) = (A_h * d_rh^(-gamma)) / Σ(A_h' * d_rh'^(-gamma))
           
        4. 결과 저장:
           CREATE TABLE nedis_meta.hospital_choice_prob (
               pat_do_cd VARCHAR,
               pat_age_gr VARCHAR,
               pat_sex VARCHAR,
               emorg_cd VARCHAR,
               probability DOUBLE,
               rank INTEGER  -- 확률 순위
           );
        """
        pass
    
    def allocate_with_capacity_constraints(self, date_str):
        """
        date_str: 'YYYYMMDD' 형식
        
        1. daily_volumes에서 해당 날짜 로드
        
        2. 각 (pat_do_cd, pat_age_gr, pat_sex) 그룹별:
           - hospital_choice_prob에서 확률 로드
           - np.random.multinomial로 병원별 초기 할당
           
        3. 용량 제약 체크:
           capacity = daily_capacity_mean + 2 * daily_capacity_std
           overflow = max(0, allocated - capacity)
           
        4. Overflow 재분배:
           - 권역센터 → 지역센터
           - 지역센터 → 지역기관
           - 같은 pat_do_cd 내에서만 재분배
           
        5. 결과 저장:
           CREATE TABLE nedis_synthetic.hospital_allocations (
               vst_dt VARCHAR,
               emorg_cd VARCHAR,
               pat_do_cd VARCHAR,
               pat_age_gr VARCHAR,
               pat_sex VARCHAR,
               allocated_count INTEGER,
               overflow_received INTEGER DEFAULT 0
           );
        """
        pass
```

### Task 3.2: IPF (Iterative Proportional Fitting) 보정
```python
# src/allocation/ipf_adjuster.py 생성

class IPFMarginalAdjuster:
    def __init__(self, conn, max_iterations=100, tolerance=0.001):
        self.conn = conn
        self.max_iter = max_iterations
        self.tol = tolerance
        
    def adjust_to_margins(self, date_str):
        """
        1. 현재 할당 로드:
           SELECT * FROM nedis_synthetic.hospital_allocations WHERE vst_dt = ?
           
        2. 타겟 마진 로드:
           SELECT * FROM nedis_synthetic.daily_volumes WHERE vst_dt = ?
           
        3. IPF 알고리즘:
           for iteration in range(self.max_iter):
               # Row adjustment (pat_do_cd × pat_age_gr × pat_sex)
               row_sums = current.groupby(['pat_do_cd', 'pat_age_gr', 'pat_sex']).sum()
               row_factors = target_margins / row_sums
               current *= row_factors
               
               # Column adjustment (emorg_cd capacity)
               col_sums = current.groupby('emorg_cd').sum()
               col_factors = hospital_capacities / col_sums
               current *= col_factors
               
               # Check convergence
               margin_error = abs(current.sum() - target).mean()
               if margin_error < self.tol:
                   break
                   
        4. 정수화 (controlled rounding):
           - 소수 부분을 확률로 사용하여 stochastic rounding
           
        5. hospital_allocations 테이블 UPDATE
        """
        pass
```

---

## Phase 4: 임상 속성 생성

### Task 4.1: DAG 기반 순차 생성기
```python
# src/clinical/dag_generator.py 생성

class ClinicalDAGGenerator:
    def __init__(self, conn):
        self.conn = conn
        self.dag_order = [
            'vst_meth',      # 내원수단
            'msypt',         # 주증상
            'ktas_fstu',     # KTAS 등급
            'main_trt_p',    # 주요치료과
            'emtrt_rust'     # 응급치료결과
        ]
        
    def generate_clinical_attributes(self, allocation_record):
        """
        allocation_record: {vst_dt, emorg_cd, pat_do_cd, pat_age_gr, pat_sex, allocated_count}
        
        1. 각 allocated_count만큼 개별 레코드 생성
        
        2. DAG 순서대로 속성 생성:
        
        # 내원수단 (vst_meth)
        query = '''
        SELECT vst_meth, COUNT(*) as freq
        FROM nedis_original.nedis2017
        WHERE pat_age_gr = ? AND pat_sex = ? AND emorg_cd = ?
        GROUP BY vst_meth
        '''
        vst_meth = self.sample_categorical(query_result)
        
        # KTAS (조건부)
        query = '''
        SELECT ktas_fstu, probability
        FROM nedis_meta.ktas_conditional_prob
        WHERE pat_age_gr = ? AND pat_sex = ? AND gubun = ? AND vst_meth = ?
        '''
        ktas_fstu = self.sample_categorical(query_result)
        
        # 응급치료결과 (emtrt_rust)
        # 11:귀가, 31:병실입원, 32:중환자실입원, 14:자의퇴실, 41:사망
        if ktas_fstu == '1':
            probs = {'32': 0.3, '31': 0.5, '41': 0.15, '11': 0.05}
        elif ktas_fstu == '5':
            probs = {'11': 0.95, '31': 0.04, '14': 0.01}
        # ... KTAS별 확률 정의
        
        3. 생성된 레코드 저장:
        CREATE TABLE nedis_synthetic.clinical_records (
            index_key VARCHAR PRIMARY KEY,  -- 생성: emorg_cd_patno_vst_dt_vst_tm
            emorg_cd VARCHAR,
            pat_reg_no VARCHAR,            -- 익명 ID 생성
            vst_dt VARCHAR,
            vst_tm VARCHAR,                 -- HHMM 랜덤 생성
            pat_age_gr VARCHAR,
            pat_sex VARCHAR,
            vst_meth VARCHAR,
            ktas_fstu VARCHAR,
            ktas01 INTEGER,                 -- ktas_fstu와 동일
            msypt VARCHAR,
            main_trt_p VARCHAR,
            emtrt_rust VARCHAR,
            -- 나머지 필드는 NULL 또는 기본값
        );
        """
        pass
```

### Task 4.2: 진단 코드 생성기
```python
# src/clinical/diagnosis_generator.py 생성

class DiagnosisGenerator:
    def __init__(self, conn):
        self.conn = conn
        
    def generate_er_diagnoses(self, clinical_record):
        """
        clinical_record: 생성된 임상 레코드
        
        1. 주진단 생성 (position = 1):
        query = '''
        SELECT diagnosis_code, probability
        FROM nedis_meta.diagnosis_conditional_prob
        WHERE pat_age_gr = ? AND pat_sex = ? AND ktas_fstu = ?
          AND is_primary = true
        ORDER BY probability DESC
        LIMIT 100
        '''
        
        2. 진단 개수 결정:
        - KTAS 1-2: 1-3개 진단
        - KTAS 3-4: 1-2개 진단  
        - KTAS 5: 1개 진단
        
        3. 부진단 생성 (position > 1):
        - 주진단과 연관된 코드 우선
        - ICD 대분류가 같은 코드에서 선택
        
        4. 저장:
        CREATE TABLE nedis_synthetic.diag_er (
            index_key VARCHAR,
            position INTEGER,
            diagnosis_code VARCHAR,
            diagnosis_category VARCHAR DEFAULT '1',
            PRIMARY KEY (index_key, position)
        );
        """
        pass
        
    def generate_admission_diagnoses(self, clinical_record):
        """
        emtrt_rust가 31(병실입원) 또는 32(중환자실입원)인 경우만 실행
        
        1. 입원 진단은 ER 진단과 70% 일치, 30% 새로운 진단
        
        2. 평균 3-5개 진단 생성
        
        3. 저장:
        CREATE TABLE nedis_synthetic.diag_adm (
            index_key VARCHAR,
            position INTEGER,
            diagnosis_code VARCHAR,
            diagnosis_category VARCHAR DEFAULT '1'
        );
        """
        pass
```

---

## Phase 5: 시간 변수 및 체류시간 생성

### Task 5.1: 체류시간 모델링
```python
# src/temporal/duration_generator.py 생성

class DurationGenerator:
    def __init__(self, conn):
        self.conn = conn
        
    def generate_er_durations(self, clinical_record):
        """
        1. KTAS별 체류시간 분포 파라미터:
        ktas_duration_params = {
            '1': {'mean': 180, 'std': 120, 'max': 720},   # 분 단위
            '2': {'mean': 240, 'std': 150, 'max': 960},
            '3': {'mean': 200, 'std': 100, 'max': 720},
            '4': {'mean': 150, 'std': 80, 'max': 480},
            '5': {'mean': 90, 'std': 40, 'max': 360}
        }
        
        2. 혼합 로그정규 분포:
        - 80%: 정규 체류
        - 20%: 장기 체류 (평균 × 2)
        
        duration_minutes = np.random.lognormal(
            mean=np.log(params['mean']),
            sigma=0.5
        )
        duration_minutes = min(duration_minutes, params['max'])
        
        3. 시간 계산:
        arrival_time = datetime.strptime(vst_dt + vst_tm, '%Y%m%d%H%M')
        discharge_time = arrival_time + timedelta(minutes=duration_minutes)
        
        otrm_dt = discharge_time.strftime('%Y%m%d')
        otrm_tm = discharge_time.strftime('%H%M')
        
        4. clinical_records 테이블 UPDATE:
        UPDATE nedis_synthetic.clinical_records
        SET otrm_dt = ?, otrm_tm = ?
        WHERE index_key = ?
        """
        pass
        
    def generate_admission_durations(self, clinical_record):
        """
        emtrt_rust가 31 또는 32인 경우만 실행
        
        1. 입원 기간 분포:
        admission_duration_params = {
            '31': {'mean': 7, 'std': 5, 'max': 30},    # 일반 병실 (일 단위)
            '32': {'mean': 14, 'std': 10, 'max': 90}   # 중환자실
        }
        
        2. Zero-inflated 음이항 분포 사용
        
        3. 입원 시간 = ER 퇴실 시간 + 30-60분
        
        4. UPDATE:
        - inpat_dt, inpat_tm: 입원 시작
        - inpat_rust: 입원 결과 (1:완쾌, 2:호전, 3:미호전, 4:사망)
        """
        pass
```

### Task 5.2: Vital Signs 생성
```python
# src/clinical/vitals_generator.py 생성

class VitalSignsGenerator:
    def __init__(self, conn):
        self.conn = conn
        
    def generate_vital_signs(self, clinical_record):
        """
        KTAS와 연령대 기반 vital signs 생성
        
        1. 측정 여부 결정:
        measurement_prob = {
            '1': 1.0,   # KTAS 1은 100% 측정
            '2': 0.95,
            '3': 0.85,
            '4': 0.70,
            '5': 0.50
        }
        
        2. 정상 범위 및 KTAS별 이상 확률:
        
        # 수축기 혈압 (vst_sbp)
        if measured:
            if ktas_fstu == '1':
                # 30% 저혈압, 20% 고혈압
                if np.random.rand() < 0.3:
                    vst_sbp = np.random.normal(85, 10)  # 저혈압
                elif np.random.rand() < 0.2:
                    vst_sbp = np.random.normal(180, 20)  # 고혈압
                else:
                    vst_sbp = np.random.normal(120, 15)  # 정상
            # ... 다른 KTAS 레벨
            
            vst_sbp = int(np.clip(vst_sbp, 60, 250))
        else:
            vst_sbp = -1
            
        # 이완기 혈압 (vst_dbp)
        vst_dbp = vst_sbp * 0.6 + np.random.normal(0, 5) if vst_sbp > 0 else -1
        
        # 맥박 (vst_per_pu)
        # 체온 (vst_bdht)
        # 호흡수 (vst_per_br)
        # 산소포화도 (vst_oxy)
        
        3. UPDATE clinical_records:
        SET vst_sbp = ?, vst_dbp = ?, vst_per_pu = ?, 
            vst_per_br = ?, vst_bdht = ?, vst_oxy = ?
        """
        pass
```

---

## Phase 6: 검증 및 품질 관리

### Task 6.1: 통계적 검증
```python
# src/validation/statistical_validator.py 생성

class StatisticalValidator:
    def __init__(self, conn):
        self.conn = conn
        
    def validate_distributions(self):
        """
        1. KS 검정 (연속 변수):
        continuous_vars = ['vst_sbp', 'vst_dbp', 'vst_per_pu']
        
        for var in continuous_vars:
            original = self.conn.execute(f'''
                SELECT {var} FROM nedis_original.nedis2017 
                WHERE {var} != -1
            ''').fetchdf()
            
            synthetic = self.conn.execute(f'''
                SELECT {var} FROM nedis_synthetic.clinical_records 
                WHERE {var} != -1
            ''').fetchdf()
            
            ks_stat, p_value = scipy.stats.ks_2samp(
                original[var], synthetic[var]
            )
            
            assert p_value > 0.05, f"KS test failed for {var}"
            
        2. Chi-square 검정 (범주형 변수):
        categorical_vars = ['ktas_fstu', 'emtrt_rust', 'pat_sex']
        
        3. 상관관계 매트릭스 비교:
        corr_original = original[numeric_cols].corr()
        corr_synthetic = synthetic[numeric_cols].corr()
        corr_diff = abs(corr_original - corr_synthetic).mean().mean()
        
        assert corr_diff < 0.05, "Correlation difference too large"
        
        4. 결과 저장:
        CREATE TABLE nedis_meta.validation_results (
            test_name VARCHAR,
            variable VARCHAR,
            statistic DOUBLE,
            p_value DOUBLE,
            passed BOOLEAN,
            test_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        pass
```

### Task 6.2: 임상 규칙 검증
```python
# src/validation/clinical_validator.py 생성

class ClinicalRuleValidator:
    def __init__(self, conn):
        self.conn = conn
        self.rules = self.load_rules()
        
    def load_rules(self):
        """
        config/clinical_rules.json 로드:
        {
            "age_diagnosis_rules": [
                {"condition": "pat_age_gr = '01' AND diagnosis_code LIKE 'I21%'", 
                 "action": "reject", 
                 "reason": "MI impossible in infants"},
                {"condition": "pat_sex = 'M' AND diagnosis_code LIKE 'O%'", 
                 "action": "reject", 
                 "reason": "Pregnancy codes for males"}
            ],
            "time_consistency_rules": [
                {"condition": "vst_dt > otrm_dt", 
                 "action": "reject", 
                 "reason": "Discharge before arrival"},
                {"condition": "DATEDIFF('minute', vst_dt||vst_tm, otrm_dt||otrm_tm) > 2880", 
                 "action": "warning", 
                 "reason": "ER stay > 48 hours"}
            ],
            "ktas_outcome_rules": [
                {"condition": "ktas_fstu = '1' AND emtrt_rust = '11'", 
                 "action": "warning", 
                 "reason": "KTAS 1 discharged home"}
            ]
        }
        """
        pass
        
    def validate_all_records(self):
        """
        1. 각 규칙별 위반 체크:
        
        for rule in self.rules:
            violations = self.conn.execute(f'''
                SELECT COUNT(*) as violation_count
                FROM nedis_synthetic.clinical_records
                WHERE {rule['condition']}
            ''').fetchone()[0]
            
            if rule['action'] == 'reject':
                assert violations == 0, f"Rule violated: {rule['reason']}"
            elif rule['action'] == 'warning':
                if violations > 0:
                    print(f"Warning: {violations} records violate: {rule['reason']}")
                    
        2. 결과 기록:
        INSERT INTO nedis_meta.validation_results
        """
        pass
```

### Task 6.3: Privacy 검증
```python
# src/validation/privacy_validator.py 생성

class PrivacyValidator:
    def __init__(self, conn):
        self.conn = conn
        
    def calculate_nearest_neighbor_distance(self, sample_size=10000):
        """
        1. 원본과 합성 데이터 샘플링:
        
        original_sample = self.conn.execute('''
            SELECT pat_age_gr, pat_sex, ktas_fstu, emtrt_rust, 
                   vst_sbp, vst_dbp, vst_per_pu
            FROM nedis_original.nedis2017
            USING SAMPLE {sample_size}
        ''').fetchdf()
        
        synthetic_sample = self.conn.execute('''
            SELECT pat_age_gr, pat_sex, ktas_fstu, emtrt_rust,
                   vst_sbp, vst_dbp, vst_per_pu
            FROM nedis_synthetic.clinical_records
            USING SAMPLE {sample_size}
        ''').fetchdf()
        
        2. Gower distance 계산 (mixed types):
        from sklearn.neighbors import NearestNeighbors
        
        3. 5th percentile distance 계산:
        distances = []
        for syn_record in synthetic_sample:
            nn_distance = find_nearest_in_original(syn_record)
            distances.append(nn_distance)
            
        p5_distance = np.percentile(distances, 5)
        
        # 임계값: 전체 거리 분포의 IQR × 1.5
        threshold = 1.5 * (np.percentile(distances, 75) - np.percentile(distances, 25))
        
        assert p5_distance > threshold, "Privacy risk: Records too similar"
        """
        pass
        
    def membership_inference_attack_test(self):
        """
        1. Shadow model 학습
        2. Attack model 학습
        3. AUC 계산
        
        목표: AUC < 0.55 (랜덤과 유사)
        """
        pass
```

---

## Phase 7: 최적화 및 캘리브레이션

### Task 7.1: 베이지안 최적화
```python
# src/optimization/bayesian_optimizer.py 생성

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

class SyntheticDataOptimizer:
    def __init__(self, conn):
        self.conn = conn
        self.search_space = {
            'gravity_gamma': Real(1.0, 2.5),
            'ipf_tolerance': Real(0.0001, 0.01),
            'dirichlet_alpha': Real(0.1, 2.0),
            'ktas_smoothing': Real(0.5, 2.0),
            'duration_sigma': Real(0.3, 0.7)
        }
        
    def objective_function(self, params):
        """
        다목적 최적화 함수:
        
        1. 파라미터로 데이터 생성 (10000 샘플)
        
        2. 메트릭 계산:
        - fidelity: KS test p-values 평균
        - utility: TSTR AUC (입원 예측 태스크)
        - privacy: Nearest neighbor distance
        - clinical: Rule violation rate
        
        3. 가중 합:
        score = (
            0.3 * fidelity +
            0.3 * utility +
            0.2 * (1 - privacy_risk) +
            0.2 * (1 - violation_rate)
        )
        
        return -score  # 최소화를 위해 음수
        """
        pass
        
    def optimize(self, n_calls=50):
        """
        optimizer = BayesSearchCV(
            estimator=self.objective_function,
            search_spaces=self.search_space,
            n_iter=n_calls,
            cv=3
        )
        
        최적 파라미터를 config/generation_params.yaml에 저장
        """
        pass
```

---

## 최종 파이프라인 실행

### Main Orchestrator
```python
# main.py 생성

import duckdb
from pathlib import Path
import yaml

class NEDISSyntheticDataPipeline:
    def __init__(self, config_path='config/generation_params.yaml'):
        self.config = yaml.safe_load(open(config_path))
        self.conn = duckdb.connect('nedis.duckdb')
        
    def run_full_pipeline(self, target_records=9_200_000):
        """
        실행 순서:
        
        1. 데이터 프로파일링 (Phase 1)
           - PopulationProfiler.extract_population_margins()
           - ConditionalProbabilityExtractor.create_all_tables()
           
        2. 볼륨 생성 (Phase 2)
           - PopulationVolumeGenerator.generate_yearly_volumes(target_records)
           - NHPPTemporalGenerator.generate_daily_events()
           
        3. 병원 할당 (Phase 3)
           - HospitalGravityAllocator.calculate_allocation_probabilities()
           - For each date:
               - allocate_with_capacity_constraints()
               - IPFMarginalAdjuster.adjust_to_margins()
               
        4. 임상 속성 생성 (Phase 4)
           - ClinicalDAGGenerator.generate_clinical_attributes()
           - DiagnosisGenerator.generate_all_diagnoses()
           
        5. 시간 변수 생성 (Phase 5)
           - DurationGenerator.generate_all_durations()
           - VitalSignsGenerator.generate_all_vitals()
           
        6. 검증 (Phase 6)
           - StatisticalValidator.validate_distributions()
           - ClinicalRuleValidator.validate_all_records()
           - PrivacyValidator.calculate_nearest_neighbor_distance()
           
        7. 최적화 (Phase 7) - Optional
           - SyntheticDataOptimizer.optimize() if performance not satisfactory
           
        8. 최종 출력
           - Export to parquet: nedis_synthetic.clinical_records
           - Generate quality report: reports/synthesis_report.html
        """
        pass

if __name__ == "__main__":
    pipeline = NEDISSyntheticDataPipeline()
    pipeline.run_full_pipeline()
```

---

## 체크포인트 및 모니터링

### 진행상황 추적 테이블
```sql
CREATE TABLE nedis_meta.pipeline_progress (
    step_name VARCHAR PRIMARY KEY,
    status VARCHAR CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    records_processed INTEGER,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    error_message TEXT,
    quality_score DOUBLE
);

-- 각 단계 완료 후 UPDATE
UPDATE nedis_meta.pipeline_progress 
SET status = 'completed', 
    end_time = CURRENT_TIMESTAMP,
    records_processed = ?
WHERE step_name = ?;
```

모든 작업은 트랜잭션으로 묶어 실패 시 롤백 가능하도록 구현하고, 각 단계별 중간 결과를 체크포인트로 저장하여 재시작 가능하게 만들 것.