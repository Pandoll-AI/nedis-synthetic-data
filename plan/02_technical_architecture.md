# NEDIS 합성 데이터 생성 기술 아키텍처

## 시스템 아키텍처 개요

### 전체 아키텍처
```
┌─────────────────────────────────────────────────────────────┐
│                    NEDIS Synthetic Data Pipeline             │
├─────────────────────────────────────────────────────────────┤
│  Configuration Layer                                         │
│  ├── generation_params.yaml                                  │
│  ├── clinical_rules.json                                     │
│  └── hospital_metadata.json                                  │
├─────────────────────────────────────────────────────────────┤
│  Processing Pipeline                                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   Phase 1   │ │   Phase 2   │ │   Phase 3   │           │
│  │  Profiling  │ │ Population  │ │ Allocation  │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   Phase 4   │ │   Phase 5   │ │   Phase 6   │           │
│  │  Clinical   │ │  Temporal   │ │ Validation  │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
├─────────────────────────────────────────────────────────────┤
│  Data Layer (DuckDB)                                        │
│  ├── nedis_original (읽기 전용)                             │
│  ├── nedis_meta (메타데이터, 통계)                          │
│  └── nedis_synthetic (생성 결과)                            │
├─────────────────────────────────────────────────────────────┤
│  Quality Assurance                                          │
│  ├── Statistical Validation                                 │
│  ├── Clinical Rule Validation                               │
│  ├── Privacy Protection                                     │
│  └── Performance Monitoring                                 │
└─────────────────────────────────────────────────────────────┘
```

## 데이터베이스 아키텍처

### DuckDB 스키마 설계

#### 1. `nedis_original` 스키마 (읽기 전용)
```sql
-- 원본 NEDIS 2017 데이터
CREATE SCHEMA IF NOT EXISTS nedis_original;

-- 메인 테이블
CREATE TABLE nedis_original.nedis2017 (
    -- 환자 정보
    pat_reg_no VARCHAR,
    pat_age_gr VARCHAR,
    pat_sex VARCHAR,
    pat_do_cd VARCHAR,
    
    -- 병원 정보
    emorg_cd VARCHAR,
    hospname VARCHAR,
    gubun VARCHAR,
    
    -- 방문 정보
    vst_dt VARCHAR,
    vst_tm VARCHAR,
    vst_meth VARCHAR,
    
    -- 임상 정보
    ktas_fstu VARCHAR,
    ktas01 INTEGER,
    msypt VARCHAR,
    
    -- 치료 정보
    main_trt_p VARCHAR,
    emtrt_rust VARCHAR,
    
    -- 퇴실 정보
    otrm_dt VARCHAR,
    otrm_tm VARCHAR,
    
    -- 생체징후
    vst_sbp INTEGER,
    vst_dbp INTEGER,
    vst_per_pu INTEGER,
    vst_per_br INTEGER,
    vst_bdht DECIMAL(4,1),
    vst_oxy INTEGER,
    
    -- 입원 정보
    inpat_dt VARCHAR,
    inpat_tm VARCHAR,
    inpat_rust VARCHAR
);

-- 진단 테이블
CREATE TABLE nedis_original.diag_er (
    index_key VARCHAR,
    position INTEGER,
    diagnosis_code VARCHAR,
    diagnosis_category VARCHAR,
    PRIMARY KEY (index_key, position)
);

CREATE TABLE nedis_original.diag_adm (
    index_key VARCHAR,
    position INTEGER,
    diagnosis_code VARCHAR,
    diagnosis_category VARCHAR,
    PRIMARY KEY (index_key, position)
);
```

#### 2. `nedis_meta` 스키마 (메타데이터 및 통계)
```sql
CREATE SCHEMA IF NOT EXISTS nedis_meta;

-- 인구학적 마진
CREATE TABLE nedis_meta.population_margins (
    pat_do_cd VARCHAR,
    pat_age_gr VARCHAR,
    pat_sex VARCHAR,
    yearly_visits INTEGER,
    seasonal_weight_spring DOUBLE,
    seasonal_weight_summer DOUBLE,
    seasonal_weight_fall DOUBLE,
    seasonal_weight_winter DOUBLE,
    weekday_weight DOUBLE,
    weekend_weight DOUBLE,
    PRIMARY KEY (pat_do_cd, pat_age_gr, pat_sex)
);

-- 병원 용량 정보
CREATE TABLE nedis_meta.hospital_capacity (
    emorg_cd VARCHAR PRIMARY KEY,
    hospname VARCHAR NOT NULL,
    gubun VARCHAR NOT NULL,
    adr VARCHAR NOT NULL,
    daily_capacity_mean INTEGER,
    daily_capacity_std INTEGER,
    ktas1_capacity INTEGER,
    ktas2_capacity INTEGER,
    attractiveness_score DOUBLE
);

-- 거리 매트릭스
CREATE TABLE nedis_meta.distance_matrix (
    from_do_cd VARCHAR,
    to_emorg_cd VARCHAR,
    distance_km DOUBLE,
    travel_time_min DOUBLE,
    PRIMARY KEY (from_do_cd, to_emorg_cd)
);

-- 병원 선택 확률
CREATE TABLE nedis_meta.hospital_choice_prob (
    pat_do_cd VARCHAR,
    pat_age_gr VARCHAR,
    pat_sex VARCHAR,
    emorg_cd VARCHAR,
    probability DOUBLE,
    rank INTEGER,
    PRIMARY KEY (pat_do_cd, pat_age_gr, pat_sex, emorg_cd)
);

-- 조건부 확률 테이블들
CREATE TABLE nedis_meta.ktas_conditional_prob (
    pat_age_gr VARCHAR,
    pat_sex VARCHAR,
    gubun VARCHAR,
    vst_meth VARCHAR,
    ktas_fstu VARCHAR,
    probability DOUBLE,
    sample_count INTEGER,
    PRIMARY KEY (pat_age_gr, pat_sex, gubun, vst_meth, ktas_fstu)
);

CREATE TABLE nedis_meta.diagnosis_conditional_prob (
    pat_age_gr VARCHAR,
    pat_sex VARCHAR,
    gubun VARCHAR,
    ktas_fstu VARCHAR,
    diagnosis_code VARCHAR,
    probability DOUBLE,
    is_primary BOOLEAN,
    sample_count INTEGER,
    PRIMARY KEY (pat_age_gr, pat_sex, gubun, ktas_fstu, diagnosis_code, is_primary)
);

-- 파이프라인 진행 상황 추적
CREATE TABLE nedis_meta.pipeline_progress (
    step_name VARCHAR PRIMARY KEY,
    status VARCHAR CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    records_processed INTEGER,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    error_message TEXT,
    quality_score DOUBLE
);

-- 검증 결과
CREATE TABLE nedis_meta.validation_results (
    test_name VARCHAR,
    variable VARCHAR,
    statistic DOUBLE,
    p_value DOUBLE,
    passed BOOLEAN,
    test_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (test_name, variable, test_timestamp)
);

-- 강화학습 훈련 로그
CREATE TABLE nedis_meta.rl_training_log (
    episode INTEGER,
    reward DOUBLE,
    ks_pass_rate DOUBLE,
    chi2_pass_rate DOUBLE,
    corr_diff DOUBLE,
    clinical_violation_rate DOUBLE,
    privacy_score DOUBLE,
    generation_time_seconds INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (episode)
);

-- 가중치 변화 추적
CREATE TABLE nedis_meta.weight_history (
    episode INTEGER,
    weight_type VARCHAR,  -- 'seasonal', 'gravity', 'ktas', etc.
    region VARCHAR,
    old_value DOUBLE,
    new_value DOUBLE,
    change_ratio DOUBLE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 최적 가중치 저장
CREATE TABLE nedis_meta.best_weights (
    LIKE nedis_meta.population_margins
);

-- 최적 파라미터 저장
CREATE TABLE nedis_meta.optimization_params (
    param_name VARCHAR,
    region VARCHAR,
    value DOUBLE,
    param_type VARCHAR DEFAULT 'global',
    optimization_method VARCHAR, -- 'bayesian' or 'reinforcement_learning'
    PRIMARY KEY (param_name, region)
);
```

#### 3. `nedis_synthetic` 스키마 (생성 결과)
```sql
CREATE SCHEMA IF NOT EXISTS nedis_synthetic;

-- 연간 볼륨
CREATE TABLE nedis_synthetic.yearly_volumes (
    pat_do_cd VARCHAR,
    pat_age_gr VARCHAR,
    pat_sex VARCHAR,
    synthetic_yearly_count INTEGER,
    generation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (pat_do_cd, pat_age_gr, pat_sex)
);

-- 일별 볼륨
CREATE TABLE nedis_synthetic.daily_volumes (
    vst_dt VARCHAR,
    pat_do_cd VARCHAR,
    pat_age_gr VARCHAR,
    pat_sex VARCHAR,
    synthetic_daily_count INTEGER,
    lambda_value DOUBLE,
    PRIMARY KEY (vst_dt, pat_do_cd, pat_age_gr, pat_sex)
);

-- 병원 할당
CREATE TABLE nedis_synthetic.hospital_allocations (
    vst_dt VARCHAR,
    emorg_cd VARCHAR,
    pat_do_cd VARCHAR,
    pat_age_gr VARCHAR,
    pat_sex VARCHAR,
    allocated_count INTEGER,
    overflow_received INTEGER DEFAULT 0,
    PRIMARY KEY (vst_dt, emorg_cd, pat_do_cd, pat_age_gr, pat_sex)
);

-- 최종 임상 레코드
CREATE TABLE nedis_synthetic.clinical_records (
    index_key VARCHAR PRIMARY KEY,
    emorg_cd VARCHAR,
    pat_reg_no VARCHAR,
    vst_dt VARCHAR,
    vst_tm VARCHAR,
    pat_age_gr VARCHAR,
    pat_sex VARCHAR,
    pat_do_cd VARCHAR,
    vst_meth VARCHAR,
    ktas_fstu VARCHAR,
    ktas01 INTEGER,
    msypt VARCHAR,
    main_trt_p VARCHAR,
    emtrt_rust VARCHAR,
    otrm_dt VARCHAR,
    otrm_tm VARCHAR,
    vst_sbp INTEGER DEFAULT -1,
    vst_dbp INTEGER DEFAULT -1,
    vst_per_pu INTEGER DEFAULT -1,
    vst_per_br INTEGER DEFAULT -1,
    vst_bdht DECIMAL(4,1) DEFAULT -1.0,
    vst_oxy INTEGER DEFAULT -1,
    inpat_dt VARCHAR,
    inpat_tm VARCHAR,
    inpat_rust VARCHAR,
    generation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 합성 진단 테이블
CREATE TABLE nedis_synthetic.diag_er (
    index_key VARCHAR,
    position INTEGER,
    diagnosis_code VARCHAR,
    diagnosis_category VARCHAR DEFAULT '1',
    PRIMARY KEY (index_key, position)
);

CREATE TABLE nedis_synthetic.diag_adm (
    index_key VARCHAR,
    position INTEGER,
    diagnosis_code VARCHAR,
    diagnosis_category VARCHAR DEFAULT '1',
    PRIMARY KEY (index_key, position)
);
```

## 소프트웨어 아키텍처

### 모듈 구조
```
src/
├── core/
│   ├── __init__.py
│   ├── config.py           # 설정 관리
│   ├── database.py         # DuckDB 연결 관리
│   ├── logger.py           # 로깅 시스템
│   └── pipeline.py         # 파이프라인 오케스트레이터
├── population/
│   ├── __init__.py
│   ├── profiler.py         # 인구학적 프로파일링
│   └── generator.py        # Dirichlet-Multinomial 생성기
├── temporal/
│   ├── __init__.py
│   ├── nhpp_generator.py   # 비균질 포아송 과정
│   └── duration_generator.py # 체류시간 모델링
├── allocation/
│   ├── __init__.py
│   ├── gravity_model.py    # 중력모형 병원 할당
│   └── ipf_adjuster.py     # IPF 마진 조정
├── clinical/
│   ├── __init__.py
│   ├── conditional_probability.py # 조건부 확률 추출
│   ├── dag_generator.py    # DAG 기반 임상 속성
│   ├── diagnosis_generator.py # 진단 코드 생성
│   └── vitals_generator.py # 생체징후 생성
├── validation/
│   ├── __init__.py
│   ├── statistical_validator.py # 통계적 검증
│   ├── clinical_validator.py    # 임상 규칙 검증
│   └── privacy_validator.py     # 프라이버시 검증
├── optimization/
│   ├── __init__.py
│   ├── bayesian_optimizer.py # 베이지안 최적화
│   ├── rl_weight_optimizer.py # 강화학습 가중치 최적화
│   └── rl_trainer.py # 강화학습 훈련 루프
└── utils/
    ├── __init__.py
    ├── math_utils.py       # 수학 유틸리티
    ├── data_utils.py       # 데이터 처리 유틸리티
    └── visualization.py    # 시각화 도구
```

### 핵심 클래스 설계

#### 1. 파이프라인 오케스트레이터
```python
class NEDISSyntheticDataPipeline:
    def __init__(self, config_path: str):
        self.config = ConfigManager(config_path)
        self.db = DatabaseManager(self.config.db_path)
        self.logger = LoggerManager()
        self.progress_tracker = ProgressTracker(self.db)
    
    async def run_full_pipeline(self, target_records: int) -> bool:
        """전체 파이프라인 실행"""
        phases = [
            ProfilerPhase(self.db, self.config),
            PopulationPhase(self.db, self.config),
            AllocationPhase(self.db, self.config),
            ClinicalPhase(self.db, self.config),
            TemporalPhase(self.db, self.config),
            ValidationPhase(self.db, self.config),
            OptimizationPhase(self.db, self.config)
        ]
        
        for phase in phases:
            success = await self.execute_phase(phase)
            if not success:
                return False
        
        return True
```

#### 2. 설정 관리자
```python
class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """YAML/JSON 설정 파일 로드"""
        
    def get_phase_config(self, phase_name: str) -> Dict[str, Any]:
        """특정 Phase 설정 반환"""
        
    def update_parameter(self, key: str, value: Any) -> None:
        """런타임 파라미터 업데이트"""
```

#### 3. 데이터베이스 관리자
```python
class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._setup_schemas()
    
    def execute_query(self, query: str, params: Optional[List] = None) -> Any:
        """쿼리 실행 with 로깅 및 오류 처리"""
        
    def batch_insert(self, table: str, data: pd.DataFrame, 
                    batch_size: int = 10000) -> None:
        """대용량 배치 삽입"""
        
    def create_checkpoint(self, checkpoint_name: str) -> None:
        """체크포인트 생성 (트랜잭션 관리)"""
```

### 성능 최적화 아키텍처

#### 1. 메모리 관리
```python
class MemoryEfficientProcessor:
    def __init__(self, chunk_size: int = 100000):
        self.chunk_size = chunk_size
        
    def process_in_chunks(self, query: str, 
                         processor: Callable) -> Generator:
        """청크 단위 처리로 메모리 사용량 제한"""
        
    def yield_batches(self, data: pd.DataFrame) -> Generator:
        """배치 단위 데이터 반환"""
```

#### 2. 병렬 처리
```python
class ParallelProcessor:
    def __init__(self, n_workers: int = None):
        self.n_workers = n_workers or cpu_count()
        
    async def process_parallel(self, tasks: List[Task]) -> List[Result]:
        """비동기 병렬 처리"""
        
    def distribute_by_date(self, date_range: List[str]) -> List[List[str]]:
        """날짜별 작업 분산"""
```

#### 3. 캐싱 시스템
```python
class CacheManager:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.memory_cache = {}
        
    def get_cached_result(self, key: str) -> Optional[Any]:
        """캐시된 결과 조회"""
        
    def cache_result(self, key: str, result: Any, 
                    ttl: int = 3600) -> None:
        """결과 캐싱"""
```

## 데이터 플로우

### 1. Phase 1: 데이터 프로파일링
```
nedis_original.nedis2017 
    ↓ [PopulationProfiler]
nedis_meta.population_margins

nedis_original.nedis2017 
    ↓ [HospitalStatisticsExtractor] 
nedis_meta.hospital_capacity

nedis_original.{nedis2017, diag_er}
    ↓ [ConditionalProbabilityExtractor]
nedis_meta.{ktas_conditional_prob, diagnosis_conditional_prob}
```

### 2. Phase 2: 인구 및 시간 패턴
```
nedis_meta.population_margins
    ↓ [PopulationVolumeGenerator]
nedis_synthetic.yearly_volumes

nedis_synthetic.yearly_volumes + nedis_meta.population_margins
    ↓ [NHPPTemporalGenerator]
nedis_synthetic.daily_volumes
```

### 3. Phase 3: 병원 할당
```
nedis_meta.{hospital_capacity, distance_matrix}
    ↓ [HospitalGravityAllocator]
nedis_meta.hospital_choice_prob

nedis_synthetic.daily_volumes + nedis_meta.hospital_choice_prob
    ↓ [AllocationEngine]
nedis_synthetic.hospital_allocations

nedis_synthetic.hospital_allocations
    ↓ [IPFMarginalAdjuster]
nedis_synthetic.hospital_allocations (조정됨)
```

### 4. Phase 4-5: 임상 속성 및 시간 변수
```
nedis_synthetic.hospital_allocations
    ↓ [ClinicalDAGGenerator]
nedis_synthetic.clinical_records (기본 속성)

nedis_synthetic.clinical_records + nedis_meta.diagnosis_conditional_prob
    ↓ [DiagnosisGenerator]
nedis_synthetic.{clinical_records, diag_er, diag_adm} (진단 추가)

nedis_synthetic.clinical_records
    ↓ [DurationGenerator + VitalSignsGenerator]
nedis_synthetic.clinical_records (완전한 레코드)
```

## 모니터링 및 관찰성

### 1. 로깅 아키텍처
```python
# 구조화된 로깅
logger = StructuredLogger({
    'level': 'INFO',
    'format': 'json',
    'fields': ['timestamp', 'level', 'phase', 'step', 'metrics']
})

# Phase별 메트릭 로깅
def log_phase_metrics(phase: str, metrics: Dict[str, Any]):
    logger.info(f"Phase {phase} completed", extra={
        'phase': phase,
        'records_processed': metrics['count'],
        'processing_time': metrics['duration'],
        'memory_usage': metrics['memory_peak'],
        'quality_score': metrics['quality']
    })
```

### 2. 실시간 모니터링
```python
class PipelineMonitor:
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alerting_system = AlertingSystem()
        
    def track_progress(self, phase: str, progress: float):
        """진행률 추적"""
        
    def check_quality_thresholds(self, metrics: Dict[str, float]):
        """품질 임계값 체크 및 알림"""
        
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """대시보드용 데이터 생성"""
```

### 3. 품질 메트릭 수집
```python
class QualityMetricsCollector:
    def collect_statistical_metrics(self) -> Dict[str, float]:
        """통계적 유사성 메트릭"""
        return {
            'ks_test_pass_rate': 0.95,
            'chi2_test_pass_rate': 0.92,
            'correlation_difference': 0.03
        }
        
    def collect_clinical_metrics(self) -> Dict[str, float]:
        """임상적 타당성 메트릭"""
        return {
            'rule_violation_rate': 0.008,
            'diagnosis_consistency': 0.97,
            'temporal_logic_errors': 2
        }
        
    def collect_privacy_metrics(self) -> Dict[str, float]:
        """프라이버시 보호 메트릭"""
        return {
            'nearest_neighbor_distance': 0.15,
            'membership_inference_auc': 0.52
        }
```

이 기술 아키텍처는 확장성, 유지보수성, 성능, 품질 보증을 모두 고려한 설계로, 대규모 의료 데이터 합성 생성 프로젝트의 복잡한 요구사항을 충족할 수 있도록 구성되었습니다.