# 3. Pattern Analysis System

## 3.1 Overview

`PatternAnalyzer` (`src/analysis/pattern_analyzer.py`)는 원본 NEDIS 데이터에서 경험적 분포를 추출하여 합성 데이터 생성의 기반을 만드는 핵심 모듈입니다. 모든 분포는 SQL 기반 집계 쿼리로 학습되며, 파라메트릭 모델(KDE, 로지스틱 회귀 등)은 사용하지 않습니다.

### 핵심 설계 원칙

- **하드코딩 금지**: 모든 확률/가중치는 원본 데이터에서 동적으로 학습
- **경험적 분포**: SQL `COUNT(*) / SUM(COUNT(*)) OVER(PARTITION BY ...)` 윈도우 함수로 조건부 확률 계산
- **계층적 Fallback**: 데이터 희소 시 상위 범주로 자동 대안
- **최소 샘플 검증**: `min_sample_size=10` 미만인 그룹은 제외
- **캐싱**: Pickle/JSON 기반, MD5 해시로 무효화 관리

---

## 3.2 Classes

### `PatternAnalysisConfig` (dataclass)

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `cache_dir` | `"cache/patterns"` | 캐시 디렉토리 |
| `use_cache` | `True` | 캐시 사용 여부 |
| `min_sample_size` | `10` | 최소 샘플 수 |
| `confidence_threshold` | `0.95` | 신뢰 수준 |
| `hierarchical_fallback` | `True` | 계층적 대안 사용 |

### `AnalysisCache`

캐시 시스템을 관리하는 클래스.

| 메서드 | 설명 |
|-------|------|
| `get_data_hash(db_manager, table_name)` | 테이블 행 수 + 1000행 샘플로 MD5 해시 생성 |
| `load_cached_analysis(analysis_type, data_hash)` | Pickle 파일에서 캐시 로드 |
| `save_analysis_cache(analysis_type, data_hash, results)` | 분석 결과를 Pickle로 저장 + JSON 메타데이터 업데이트 |
| `clear_cache(analysis_type?)` | 전체 또는 특정 분석 캐시 삭제 |

**캐시 키 구조**: `{analysis_type}_{data_hash}.pkl`
**메타데이터**: `cache/patterns/cache_metadata.json`에 각 엔트리의 생성 시각, 파일 경로 기록

### `PatternAnalyzer`

핵심 분석 클래스. `DatabaseManager`와 `ConfigManager`를 주입받아 동작합니다.

---

## 3.3 Pattern Categories (10가지)

`analyze_all_patterns()` 메서드가 순차적으로 10개 패턴을 분석합니다:

| # | 패턴 이름 | 분석 메서드 | 학습 내용 |
|---|---------|-----------|----------|
| 1 | `hospital_allocation` | `analyze_hospital_allocation_patterns()` | P(hospital \| region) |
| 2 | `ktas_distributions` | `analyze_ktas_distributions()` | P(KTAS \| region, hospital_type) — 4단계 계층 |
| 3 | `regional_patterns` | `analyze_regional_patterns()` | 지역별 기초 통계 (방문수, 응급률, 남성비 등) |
| 4 | `demographic_patterns` | `analyze_demographic_patterns()` | P(age_group, sex) 및 동반 통계 |
| 5 | `temporal_patterns` | `analyze_temporal_patterns()` | 월별/요일별/시간별 내원 분포 |
| 6 | `temporal_conditional_patterns` | `analyze_temporal_conditional_patterns()` | 8가지 다차원 조건부 시간 분포 |
| 7 | `visit_method_patterns` | `analyze_visit_method_patterns()` | P(vst_meth \| pat_age_gr) |
| 8 | `chief_complaint_patterns` | `analyze_chief_complaint_patterns()` | P(msypt \| pat_age_gr, pat_sex) |
| 9 | `department_patterns` | `analyze_department_patterns()` | P(main_trt_p \| pat_age_gr, pat_sex) |
| 10 | `treatment_result_patterns` | `analyze_treatment_result_patterns()` | P(emtrt_rust \| ktas_fstu, pat_age_gr) |

각 분석은 캐시 미스 시에만 수행되며, 결과는 즉시 Pickle로 캐싱됩니다.

---

## 3.4 SQL-Based Empirical Distribution Extraction

모든 패턴은 SQL 윈도우 함수로 조건부 확률을 계산합니다. 대표적인 패턴:

### Hospital Allocation

```sql
SELECT
    pat_do_cd as region_code,
    emorg_cd as hospital_code,
    COUNT(*) as visit_count,
    COUNT(*) * 1.0 / SUM(COUNT(*)) OVER(PARTITION BY pat_do_cd) as region_probability
FROM {source_table}
WHERE pat_do_cd IS NOT NULL AND emorg_cd IS NOT NULL
GROUP BY pat_do_cd, emorg_cd
HAVING COUNT(*) >= {min_sample_size}
```

결과: 각 지역 내에서 병원별 방문 확률을 계산하여 `np.random.choice`의 확률 벡터로 사용.

### Conditional Temporal Patterns

단일 쿼리로 6개 차원(pat_age_gr, pat_sex, ktas_fstu, month, dow, hour)의 원시 카운트를 추출한 후, `_build_conditional_distribution()`으로 8가지 조건부 분포를 구축:

| 조건부 분포 | Key Columns | Target |
|-----------|-------------|--------|
| `month_hour` | [month] | hour |
| `dow_hour` | [dow] | hour |
| `month_dow_hour` | [month, dow] | hour |
| `ktas_hour` | [ktas_fstu] | hour |
| `age_hour` | [pat_age_gr] | hour |
| `age_sex_hour` | [pat_age_gr, pat_sex] | hour |
| `ktas_age_hour` | [ktas_fstu, pat_age_gr] | hour |
| `age_sex_ktas_hour` | [pat_age_gr, pat_sex, ktas_fstu] | hour |

---

## 3.5 `_build_conditional_distribution()` Method

SQL 쿼리 결과를 조건부 확률 맵으로 변환하는 범용 메서드:

1. **입력**: `count` 컬럼을 포함한 DataFrame, 키 컬럼 리스트, 타겟 컬럼
2. **그룹화**: `key_cols + [target_col]`로 그룹화 후 count 합산
3. **정규화**: 각 키 조합 내에서 `count / total_count`로 확률 계산
4. **출력 형식**: `{key_string: {total_count, patterns: {hour: {count, probability}}}}`

키 문자열은 `'|'`로 조인 (예: `"3|M"` = ktas 3, 남성).

---

## 3.6 Hierarchical Fallback Strategy

### KTAS Distribution — 4-Level Hierarchy

`get_hierarchical_ktas_distribution(region_code, hospital_type)`:

```
Level 1: "{region_code}_{hospital_type}"      (예: "1101_large")
    ↓ miss
Level 2: "{region_code[:2]}_{hospital_type}"  (예: "11_large")
    ↓ miss
Level 3: hospital_type only                   (예: "large")
    ↓ miss
Level 4: overall_pattern                      (전체 평균)
```

각 단계에서 패턴이 존재하면 즉시 반환. 병원 유형은 `daily_capacity_mean` 기준으로 분류:
- `>= 300`: large
- `>= 100`: medium
- 나머지: small

### Hospital Allocation — Region-Based Hierarchy

`_create_hierarchical_patterns()`으로 대분류(2자리) 수준의 병원 할당 패턴 생성:
- 소분류 지역의 병원 방문 카운트를 대분류로 합산
- 합산된 카운트로 확률 재계산
- 소분류 데이터가 `min_sample_size` 미만이면 대분류 패턴으로 대체

---

## 3.7 Caching Strategy

### Hash-Based Invalidation

```python
hash_input = f"{table_name}_{row_count}_{sample_data.to_string()}"
data_hash = hashlib.md5(hash_input.encode()).hexdigest()
```

- 테이블 행 수 + 1000행 랜덤 샘플의 문자열 표현으로 MD5 해시 생성
- 원본 데이터 변경 시 해시가 달라져 자동 캐시 무효화

### Storage

- **분석 결과**: `cache/patterns/{analysis_type}_{data_hash}.pkl` (Pickle)
- **메타데이터**: `cache/patterns/cache_metadata.json` (JSON)
- **선택적 삭제**: `clear_cache(analysis_type)` — 특정 분석만 재수행 가능

### Performance

- 최초 분석: 10개 SQL 쿼리 실행 (수 초~수십 초)
- 캐시 적중: Pickle 로드만 수행 (밀리초 단위)

---

## 3.8 Source Table Configuration

소스 테이블은 다음 순서로 결정:

1. `config['original.source_table']` — 명시적 테이블명
2. `config['original.year']` — 연도 기반 (`nedis_original.nedis{year}`)
3. 기본값: `nedis_original.nedis2017`

---

## 3.9 Related Modules

- **Consumer**: `VectorizedPatientGenerator` — 분석된 패턴을 소비하여 환자 생성
- **Consumer**: `TemporalPatternAssigner` — 조건부 시간 분포를 시간 할당에 사용
- **Consumer**: HTML Generator (`templates/nedis_generator_template.html`) — 패턴을 base64로 임베딩
- **Validation**: `CorrelationBalanceValidator` — 생성된 데이터의 상관관계 검증

---
