# 4. Synthetic Data Generation

## 4.1 Overview

합성 데이터 생성 파이프라인은 3개의 핵심 모듈로 구성됩니다:

| 모듈 | 파일 | 역할 |
|------|-----|------|
| `VectorizedPatientGenerator` | `src/vectorized/patient_generator.py` | 환자 속성 벡터화 생성 (날짜 없이) |
| `TemporalPatternAssigner` | `src/vectorized/temporal_assigner.py` | 날짜/시간 할당 |
| `CapacityConstraintPostProcessor` | `src/vectorized/capacity_processor.py` | 병원 용량 제약 후처리 |

**핵심 아키텍처**: 시간 분리 전략 — 먼저 전체 환자 속성을 한 번에 생성하고, 이후 날짜/시간을 별도 할당합니다. 이를 통해 날짜별 반복 생성 대비 **약 50배 성능 향상**을 달성합니다.

---

## 4.2 VectorizedPatientGenerator

### 4.2.1 Configuration

```python
@dataclass
class PatientGenerationConfig:
    total_records: int = 322573    # 목표 레코드 수
    batch_size: int = 50000        # 메모리 최적화 청크 크기
    random_seed: Optional[int] = None
    memory_efficient: bool = True
```

### 4.2.2 4-Stage Generation Pipeline

`_generate_patients_vectorized(total_records)` 내부의 4단계:

#### Stage 1: Demographics (`_generate_demographics_vectorized`)

`np.random.multinomial`을 사용하여 `(pat_do_cd, pat_age_gr, pat_sex)` 동시 생성:

1. PatternAnalyzer의 `demographic_patterns`에서 `(age_group, sex)` 조합별 확률 로드
2. `np.random.multinomial(total_records, probabilities)` — 전체 레코드를 조합별로 한 번에 분배
3. 각 조합에 해당하는 레코드 수만큼 컬럼 값 할당
4. 지역 코드(`pat_do_cd`)는 `regional_patterns`의 방문 비율로 할당

#### Stage 2: Hospital Assignment (`_generate_hospital_assignments_vectorized`)

계층적 `np.random.choice`로 병원 할당:

1. PatternAnalyzer의 `hospital_allocation` 패턴에서 지역별 병원 선택 확률 로드
2. 각 지역 그룹에 대해:
   - 해당 지역의 병원 목록과 확률 벡터 조회
   - `np.random.choice(hospitals, size=group_size, p=probabilities)`
3. **Fallback**: 지역 패턴 부재 시 `hierarchical_fallback`의 대분류(2자리) 패턴 사용
4. **최종 Fallback**: 패턴 미확보 시 `_assign_random_hospitals()`

#### Stage 3: Independent Clinical Attributes (`_generate_independent_clinical_attributes`)

Demographics에만 의존하는 속성들을 완전 벡터화 생성:

| 속성 | 조건부 분포 | 소스 패턴 |
|------|-----------|----------|
| `vst_meth` (내원수단) | P(vst_meth \| pat_age_gr) | `visit_method_patterns` |
| `msypt` (주증상) | P(msypt \| pat_age_gr, pat_sex) | `chief_complaint_patterns` |
| `main_trt_p` (주요치료과) | P(main_trt_p \| pat_age_gr, pat_sex) | `department_patterns` |

각 속성은 그룹별로 `np.random.choice`를 호출하여 조건부 확률에 따라 샘플링합니다.

#### Stage 4: Conditional Clinical Attributes (`_generate_conditional_clinical_attributes`)

순차적 의존 관계가 있는 속성들의 Semi-벡터화 생성:

1. **KTAS 생성**: `get_hierarchical_ktas_distribution(region_code, hospital_type)`로 4단계 계층적 분포 조회 → `np.random.choice` 적용
   - 그룹 키: `(region_code, hospital_type)`
   - 그룹 내 일괄 샘플링
2. **치료결과 생성**: P(emtrt_rust | ktas_fstu, pat_age_gr)
   - KTAS 생성 후 그 결과를 조건으로 사용
   - 그룹 키: `(ktas_fstu, pat_age_gr)`

### 4.2.3 Chunked Processing

대용량 데이터 생성 시 메모리 효율을 위해 청크 단위 처리:

```python
def _generate_patients_chunked(gen_config):
    chunks = []
    remaining = gen_config.total_records
    while remaining > 0:
        chunk_size = min(gen_config.batch_size, remaining)
        chunk_df = self._generate_patients_vectorized(chunk_size)
        chunks.append(chunk_df)
        remaining -= chunk_size
    return pd.concat(chunks, ignore_index=True)
```

---

## 4.3 TemporalPatternAssigner

### 4.3.1 Configuration

```python
@dataclass
class TemporalConfig:
    year: int = 2017
    time_resolution: str = 'hourly'
    enable_conditional_hour_patterns: bool = True
    conditional_global_mix_weight: float = 0.15
    conditional_smoothing_alpha: float = 0.05
    conditional_context_min_count: int = 50
```

### 4.3.2 Temporal Assignment Pipeline

`assign_temporal_patterns(patients_df, temporal_config)`:

#### Step 1: Load Temporal Patterns (`_load_temporal_patterns`)

PatternAnalyzer 결과에서 `temporal_patterns`(월별/요일별/시간별)과 `temporal_conditional_patterns`(8가지 조건부) 로드.

#### Step 2: Calculate Daily Volumes (`_calculate_daily_volumes`)

각 날짜의 예상 방문량을 계산:

1. **기본 일일 방문량**: `total_records / 365`
2. **월별 계절 효과**: 월별 확률 × 12 → 승수로 변환
3. **요일 효과**: 요일별 확률 × 7 → 승수로 변환
4. **공휴일 효과**: 공휴일 당일에 감소 승수 적용
5. **Poisson 변동**: `np.random.poisson(expected)` — 일별 자연 변동 추가
6. **정규화**: 전체 합이 `total_records`와 일치하도록 스케일링

#### Step 3: Assign Dates (`_assign_dates_vectorized`)

`np.searchsorted` 기반 벡터화 날짜 할당:

1. 일별 볼륨의 누적합(CDF) 계산
2. `[0, total_records)` 범위의 균등 난수 생성
3. `np.searchsorted(cdf, random_values)` — 각 레코드를 해당 날짜에 할당
4. 날짜 문자열 형식으로 변환 (`YYYYMMDD`)

#### Step 4: Assign Times (`_assign_times_vectorized`)

조건부 시간 분포 블렌딩으로 시간 할당:

**조건부 모드** (`enable_conditional_hour_patterns=True`):
1. 환자를 `(month, dow, age, sex, ktas)` 그룹으로 분류
2. 각 그룹에 `_resolve_hour_distribution()` 호출
3. 결과 확률 벡터로 `np.random.choice(0..23, p=distribution)` 적용
4. 분(minute)은 `np.random.randint(0, 60)` 독립 생성

**단순 모드**: 전역 시간별 분포에서 직접 샘플링

### 4.3.3 Weighted Distribution Blending (`_resolve_hour_distribution`)

8개 조건부 분포 후보를 가중 블렌딩하여 최종 시간 분포를 결정:

| 후보 | Key 형식 | 가중치 |
|------|---------|-------|
| `month_dow_hour` | `{month}\|{dow}` | 0.35 |
| `month_hour` | `{month}` | 0.25 |
| `dow_hour` | `{dow}` | 0.15 |
| `age_sex_ktas_hour` | `{age}\|{sex}\|{ktas}` | 0.10 |
| `ktas_age_hour` | `{ktas}\|{age}` | 0.08 |
| `ktas_hour` | `{ktas}` | 0.07 |
| `age_hour` | `{age}` | 0.05 |
| `age_sex_hour` | `{age}\|{sex}` | 0.05 |

**블렌딩 프로세스**:
1. 각 후보에서 `total_count >= conditional_context_min_count`인 패턴만 사용
2. 해당 패턴의 24시간 확률 벡터 × 가중치를 누적
3. 누적 가중치로 나누어 정규화
4. 전역 분포와 혼합: `(1 - mix_weight) * conditional + mix_weight * global`
5. Laplace smoothing: `(1 - alpha) * blended + alpha * uniform(1/24)`
6. **Fallback**: 어떤 조건부 패턴도 없으면 전역 분포 사용

### 4.3.4 Korean Holiday Effects

`_get_korean_holidays(year)`:

고정 공휴일:
- 1/1 신정, 3/1 삼일절, 5/5 어린이날, 6/6 현충일
- 8/15 광복절, 10/3 개천절, 10/9 한글날, 12/25 성탄절

음력 공휴일 (연도별 하드코딩):
- 설날 (음력 1/1 전후 3일)
- 추석 (음력 8/15 전후 3일)
- 부처님오신날 (음력 4/8)

공휴일은 `_calculate_daily_volumes()`에서 감소 승수(기본 0.7~0.8)로 적용됩니다.

---

## 4.4 CapacityConstraintPostProcessor

### 4.4.1 Pipeline

`apply_capacity_constraints(patients_df, capacity_config)`:

1. **용량 참조 데이터 로드**: 병원별 `daily_capacity_mean` 로드
2. **동적 용량 한도 계산**:
   - 기본 용량 × 안전 여유 계수 (기본 1.2)
   - 주말: × 0.8 (기본)
   - 공휴일: × 0.7 (기본)
3. **현재 부하 계산**: 날짜+병원별 할당된 환자 수 집계
4. **Overflow 감지**: 일일 부하 > 용량 한도인 병원 식별
5. **재할당**: Overflow 환자를 여유 병원으로 이동

### 4.4.2 Redistribution Strategies

| 전략 | 설명 |
|------|------|
| `nearest_available` | 같은 지역 내 여유 병원 우선 |
| `random_available` | 전체 여유 병원 중 랜덤 선택 |
| `second_choice_probability` | 원래 지역의 차순위 병원 확률 기반 선택 |

재할당된 레코드에는 `overflow_flag`와 `redistribution_method` 메타데이터가 기록됩니다.

---

## 4.5 Performance Characteristics

| 항목 | 값 |
|------|---|
| 322K 레코드 생성 | ~7초 (벡터화) vs ~300초 (순차) |
| 메모리 사용 | ~2GB (322K 기준, 50K 청크) |
| 캐시 적중 시 패턴 로드 | 밀리초 단위 |
| 스케일링 | 수백만 레코드까지 선형 확장 가능 |

### 성능 향상 핵심 요인

1. **시간 분리**: 날짜별 반복 제거 (365회 → 1회)
2. **벡터화**: NumPy 배열 연산으로 Python 루프 대체
3. **그룹 일괄 처리**: 동일 조건 그룹을 한 번의 `np.random.choice`로 처리
4. **패턴 캐싱**: SQL 쿼리 결과 재사용

---

## 4.6 Data Flow Summary

```
PatternAnalyzer.analyze_all_patterns()
    ↓ (10가지 패턴)
VectorizedPatientGenerator._generate_patients_vectorized()
    Stage 1: Demographics  (np.random.multinomial)
    Stage 2: Hospital      (np.random.choice per region)
    Stage 3: Independent   (np.random.choice per group)
    Stage 4: Conditional   (hierarchical KTAS → treatment result)
    ↓ (DataFrame without dates)
TemporalPatternAssigner.assign_temporal_patterns()
    Step 1: Load patterns
    Step 2: Daily volumes  (seasonal × weekday × holiday × Poisson)
    Step 3: Dates          (np.searchsorted on CDF)
    Step 4: Times          (8-candidate weighted blending)
    ↓ (DataFrame with dates/times)
CapacityConstraintPostProcessor.apply_capacity_constraints()
    → Overflow detection & redistribution
    ↓ (Final DataFrame)
```

---
