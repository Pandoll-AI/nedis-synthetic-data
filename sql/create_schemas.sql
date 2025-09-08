-- NEDIS 합성 데이터 생성 시스템 스키마 정의
-- DuckDB 스키마 생성

-- 원본 데이터 스키마 (읽기 전용)
CREATE SCHEMA IF NOT EXISTS nedis_original;

-- 합성 데이터 스키마
CREATE SCHEMA IF NOT EXISTS nedis_synthetic;

-- 메타데이터 스키마
CREATE SCHEMA IF NOT EXISTS nedis_meta;

-- nedis_meta.hospital_capacity 테이블 생성
CREATE TABLE IF NOT EXISTS nedis_meta.hospital_capacity (
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
CREATE TABLE IF NOT EXISTS nedis_meta.population_margins (
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

-- 거리 매트릭스
CREATE TABLE IF NOT EXISTS nedis_meta.distance_matrix (
    from_do_cd VARCHAR,
    to_emorg_cd VARCHAR,
    distance_km DOUBLE,
    travel_time_min DOUBLE,
    PRIMARY KEY (from_do_cd, to_emorg_cd)
);

-- 병원 선택 확률
CREATE TABLE IF NOT EXISTS nedis_meta.hospital_choice_prob (
    pat_do_cd VARCHAR,
    pat_age_gr VARCHAR,
    pat_sex VARCHAR,
    emorg_cd VARCHAR,
    probability DOUBLE,
    rank INTEGER,
    PRIMARY KEY (pat_do_cd, pat_age_gr, pat_sex, emorg_cd)
);

-- KTAS 조건부 확률 테이블
CREATE TABLE IF NOT EXISTS nedis_meta.ktas_conditional_prob (
    pat_age_gr VARCHAR NOT NULL,
    pat_sex VARCHAR NOT NULL,
    gubun VARCHAR NOT NULL,           -- 병원 종별
    vst_meth VARCHAR NOT NULL,        -- 내원수단 (1:119구급차, 3:기타구급차, 6:도보 등)
    ktas_fstu VARCHAR NOT NULL,       -- KTAS 등급 (1,2,3,4,5)
    probability DOUBLE NOT NULL,
    sample_count INTEGER NOT NULL,
    PRIMARY KEY (pat_age_gr, pat_sex, gubun, vst_meth, ktas_fstu)
);

-- 진단 조건부 확률 테이블
CREATE TABLE IF NOT EXISTS nedis_meta.diagnosis_conditional_prob (
    pat_age_gr VARCHAR,
    pat_sex VARCHAR,
    gubun VARCHAR,
    ktas_fstu VARCHAR,
    diagnosis_code VARCHAR,  -- diag_er 테이블의 diagnosis_code
    probability DOUBLE,
    is_primary BOOLEAN,      -- position = 1인 경우
    sample_count INTEGER,
    PRIMARY KEY (pat_age_gr, pat_sex, gubun, ktas_fstu, diagnosis_code, is_primary)
);

-- 파이프라인 진행 상황 추적
CREATE TABLE IF NOT EXISTS nedis_meta.pipeline_progress (
    step_name VARCHAR PRIMARY KEY,
    status VARCHAR CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    records_processed INTEGER,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    error_message TEXT,
    quality_score DOUBLE
);

-- 검증 결과
CREATE TABLE IF NOT EXISTS nedis_meta.validation_results (
    test_name VARCHAR,
    variable VARCHAR,
    statistic DOUBLE,
    p_value DOUBLE,
    passed BOOLEAN,
    test_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (test_name, variable, test_timestamp)
);

-- 연간 볼륨
CREATE TABLE IF NOT EXISTS nedis_synthetic.yearly_volumes (
    pat_do_cd VARCHAR,
    pat_age_gr VARCHAR,
    pat_sex VARCHAR,
    synthetic_yearly_count INTEGER,
    generation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (pat_do_cd, pat_age_gr, pat_sex)
);

-- 일별 볼륨
CREATE TABLE IF NOT EXISTS nedis_synthetic.daily_volumes (
    vst_dt VARCHAR,
    pat_do_cd VARCHAR,
    pat_age_gr VARCHAR,
    pat_sex VARCHAR,
    synthetic_daily_count INTEGER,
    lambda_value DOUBLE,
    PRIMARY KEY (vst_dt, pat_do_cd, pat_age_gr, pat_sex)
);

-- 병원 할당
CREATE TABLE IF NOT EXISTS nedis_synthetic.hospital_allocations (
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
CREATE TABLE IF NOT EXISTS nedis_synthetic.clinical_records (
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
CREATE TABLE IF NOT EXISTS nedis_synthetic.diag_er (
    index_key VARCHAR,
    position INTEGER,
    diagnosis_code VARCHAR,
    diagnosis_category VARCHAR DEFAULT '1',
    PRIMARY KEY (index_key, position)
);

CREATE TABLE IF NOT EXISTS nedis_synthetic.diag_adm (
    index_key VARCHAR,
    position INTEGER,
    diagnosis_code VARCHAR,
    diagnosis_category VARCHAR DEFAULT '1',
    PRIMARY KEY (index_key, position)
);