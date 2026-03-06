# Revision v0.4 — Temporal Pattern Bug Fixes

Date: 2026-03-02

조건부 시간 분포(conditional temporal distributions)의 학습 및 적용 과정에서 발견된 3개 버그를 수정합니다. 이 버그들로 인해 합성 데이터의 시간대별 도착 분포가 원본 대비 평탄화되는 현상이 발생했습니다.

## Bug 1: 조건부 시간 분포 평탄화 (Critical)

**파일**: `src/analysis/pattern_analyzer.py` — `_build_conditional_distribution()`

**증상**: `month_dow_hour`, `dow_hour`, `month_hour` 등 조건부 시간 분포가 거의 균등 분포(~4.2% per hour)로 나타남. 원본의 18~21시 피크(6~7%)가 사라짐.

**원인**: SQL 쿼리는 `(pat_age_gr, pat_sex, ktas_fstu, month, dow, hour)` 6개 차원으로 GROUP BY하여 `count` 컬럼에 방문 건수를 집계. 이후 `_build_conditional_distribution()`이 `month_hour` 등 2개 차원으로 재집계할 때, `count` 컬럼의 **값을 합산(sum)** 해야 하지만 **행의 개수를 세는(count)** 잘못된 연산을 수행.

```python
# Before (bug): drops 'count' column, counts rows instead
df[group_cols]                    # 'count' column dropped
  .groupby(...)['hour'].count()   # counts rows (usually 1 per cell)

# After (fix): sums actual visit counts
df[group_cols + ['count']]        # keeps 'count' column
  .groupby(...)['count'].sum()    # sums actual visit counts
```

**결과**: 6차원 GROUP BY 결과를 2차원으로 재집계할 때, 각 `(month, hour)` 조합에 대응하는 행이 여러 개(age/sex/ktas 조합별 1행)이지만, 그 행 수는 시간대와 거의 무관하여 균등 분포가 됨.

## Bug 2: DOW 매핑 불일치 (Python generator)

**파일**: `src/vectorized/temporal_assigner.py` — `_assign_conditional_hour_patterns()`

**증상**: 조건부 시간 분포의 요일(DOW) 키와 실제 요일이 1일씩 어긋남.

**원인**:
- SQL 쿼리(`EXTRACT(DOW ...)`)는 DuckDB/PostgreSQL 규약으로 **0=일요일, 6=토요일** 사용
- `_assign_conditional_hour_patterns()`는 pandas `dayofweek`로 **0=월요일, 6=일요일** 사용
- `_calculate_daily_volumes()`는 이미 올바르게 `(dow + 1) % 7` 변환을 수행했으나, `_assign_conditional_hour_patterns()`에만 변환이 누락

```python
# Before (bug): pandas dayofweek (0=Mon, 6=Sun)
result_df['_dow'] = date_obj.dt.dayofweek

# After (fix): convert to DuckDB DOW (0=Sun, 6=Sat)
result_df['_dow'] = ((date_obj.dt.dayofweek + 1) % 7)
```

## Bug 3: DOW 매핑 불일치 (HTML generator)

**파일**: `templates/nedis_generator_template.html`

**증상**: JS 생성기의 요일 매핑이 불필요한 변환을 수행.

**원인**: `jsToPyDow = [6,0,1,2,3,4,5]`로 JS `getDay()` (0=Sun)을 Python `weekday()` (0=Mon)으로 변환했으나, 패턴 캐시는 DuckDB DOW (0=Sun) 기준. JS `getDay()`와 DuckDB DOW가 동일한 규약이므로 변환이 불필요.

```javascript
// Before (bug): unnecessary conversion
var jsToPyDow = [6, 0, 1, 2, 3, 4, 5];
dateDows[d] = jsToPyDow[dt.getDay()];

// After (fix): direct use
dateDows[d] = dt.getDay();
```

캐시 재생성 후, 조건부 분포를 재활성화하고 키 구분자를 `_` → `|`로 수정 (캐시의 실제 키 형식과 일치).

## 영향 범위

| 항목 | 수정 전 | 수정 후 |
|------|---------|---------|
| 시간대별 도착 분포 | 거의 균등 (~4.2%/h) | 원본 반영 (18~21시 6~7% 피크) |
| 요일별 시간 분포 | 1일 씩 shift된 패턴 적용 | 올바른 요일에 대응하는 패턴 적용 |
| 일별 방문량 가중치 | 정상 (이미 올바른 변환) | 변경 없음 |

## 캐시 재생성 필요

Bug 1 수정 후, 기존 `temporal_conditional_patterns_*.pkl` 캐시는 잘못된 분포를 포함하고 있으므로 **캐시를 삭제하고 재생성**해야 합니다:

```bash
rm cache/patterns/temporal_conditional_patterns_*.pkl
python scripts/generate.py --force-reanalyze  # 또는 캐시 무효화 옵션 사용
```

HTML 생성기용 patterns.json도 재추출 필요:

```bash
python scripts/extract_patterns_for_html.py
python scripts/build_html_generator.py
```

## 코드 리뷰에서 발견된 추가 수정

### Issue A: `_hour_profile_source` 컬럼 누수

**파일**: `src/vectorized/temporal_assigner.py`

`_assign_conditional_hour_patterns()`에서 디버깅용 `_hour_profile_source` 컬럼을 생성하지만, 반환 시 `drop`하지 않아 출력 DataFrame에 누수됨. drop 목록에 추가.

### Issue B: `correlation_balance_validator.py` DOW 매핑

**파일**: `src/validation/correlation_balance_validator.py`

합성 데이터의 DOW를 pandas `dayofweek` (0=Mon)로 계산하여 원본(DuckDB 0=Sun)과 비교 시 1일 shift. 동일한 `((dayofweek + 1) % 7)` 변환 적용.

### Issue C: `_build_conditional_distribution` 가드 추가

**파일**: `src/analysis/pattern_analyzer.py`

`count` 컬럼이 없는 DataFrame이 전달될 경우 명확한 에러 메시지를 출력하도록 `ValueError` 가드 추가.

## 수정된 파일

- `src/analysis/pattern_analyzer.py` — `_build_conditional_distribution()`: `.count()` → `.sum()`, `count` 컬럼 가드 추가
- `src/vectorized/temporal_assigner.py` — DOW 변환 추가, `_hour_profile_source` 컬럼 drop 추가
- `src/validation/correlation_balance_validator.py` — DOW 변환 추가
- `templates/nedis_generator_template.html` — `jsToPyDow` 제거, 조건부 분포 키 구분자 수정(`_` → `|`), 캐시 재생성 후 조건부 분포 재활성화
