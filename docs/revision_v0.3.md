# Revision v0.3 — Vectorized Pipeline Improvements

Date: 2025-09-09

This revision aligns the implementation with the vectorized generation design and removes hard-coded parameters by shifting to dynamic, data-driven distributions with config-backed fallbacks.

## Summary of Changes

- Dynamic attribute distributions:
  - Added analyzers for P(vst_meth | age), P(msypt | age, sex), P(main_trt_p | age, sex), P(emtrt_rust | KTAS, age).
  - Replaced all hard-coded sampling in patient generation with these learned patterns.

- Batch KTAS generation:
  - Rewrote KTAS assignment to operate in grouped batches by (region, hospital_type) using hierarchical distributions, removing per-row loops.

- Capacity redistribution:
  - Implemented same-region-first reassignment (no distance fallback). Removed distance-matrix usage and the nearest-within-50km option.

- Quality gate:
  - Added optional quality gate that runs Statistical and Clinical validations; pipeline fails if the combined score falls below a threshold (default 0.95).

- CLI defaults and options:
  - Batch size default set to 100,000 and memory-efficient mode enabled by default.
  - Overflow method choices updated to [random_available, same_region_first, second_choice_probability].
  - Canonical CLI: scripts/generate.py (one-file practical entry).  
    Legacy: scripts/run_vectorized_pipeline.py prints a deprecation notice; scripts/ndis_synth.py remains as a thin wrapper.

- Hard-coded assumptions removed:
  - Hourly fallback distribution moved to config (`temporal.fallback_hour_weights`).
  - Attribute fallback distributions moved to config (`fallback.distributions`).

## Touched Modules

- `src/analysis/pattern_analyzer.py`
  - New analyzers: visit_method_patterns, chief_complaint_patterns, department_patterns, treatment_result_patterns.
- `src/vectorized/patient_generator.py`
  - Independent attributes now sampled from dynamic distributions.
  - KTAS assignment now grouped/batch-based; treatment outcomes use learned conditional distributions.
- `src/vectorized/temporal_assigner.py`
  - Fallback hourly time distribution pulled from config (no hard-coding).
- `src/vectorized/capacity_processor.py`
  - Same-region-first redistribution; removed distance-based logic.
- `scripts/run_vectorized_pipeline.py`
  - CLI defaults updated; quality gate integrated; overflow method options updated.
- `src/core/config.py`
  - Added fallback distributions and temporal fallback hour weights.

## Migration Notes

- If running without original data tables, configure `fallback.distributions` and `temporal.fallback_hour_weights` in `config/generation_params.yaml` to ensure reasonable sampling.
- If you previously relied on `nearest_available`, use `same_region_first` or `second_choice_probability` instead.

## 개인정보 보호 위험 평가 (v0.3)

본 절은 v0.3 개선 사항이 개인정보 보호에 미치는 영향을 냉정하고 객관적으로 평가합니다. 결론적으로, 본 버전은 “통계적 유사성 향상”과 “하드코딩 제거”를 달성했으나, 개인정보 재식별 위험은 별도 통제를 도입하지 않는 이상 본질적으로 감소했다고 단정할 수 없습니다. 다음 위협 모델, 잠재적 공격면, 정량화 지표, 완화 전략을 제시합니다.

### 위협 모델과 공격면
- 멤버십 추론 공격 (Membership Inference, MIA): 원본 데이터에 특정 개인 레코드가 포함되었는지 여부를 합성 결과를 통해 추정.
- 속성 추론 공격 (Attribute Inference): 공개/부분 정보(예: 성별, 나이대, 방문 지역)를 기반으로 비공개 속성(예: 중증도, 치료결과)을 추정.
- 연결 재식별 (Linkage): 외부 데이터셋(언론, 보험, 병원 공개 통계 등)과의 조인을 통해 개인 또는 소집합을 재식별.
- 희귀 패턴 유출: 드문 조합(소아+특정 중증도+특정 시간대+소수 병원 등)이 합성에서 과도히 보존될 경우 사실상 준식별자 역할 수행.
- 시간·공간 유일성: 시계열 도착 패턴(분 단위 간격), 병원/지역 단위의 소세분 집계가 소수 집단을 노출.

### v0.3 변경에 따른 위험 영향
- 동적 패턴 학습 강화: 하드코딩 제거와 분포 충실도 증가는 유용성을 높이나, 원본 드문 조합을 그대로 반영할 가능성도 높임. 캐싱된 분포가 소규모 셀을 포함하면 재식별 위험이 증가합니다.
- 배치 KTAS/치료결과 샘플링: 조건부 분포의 정확도 향상은 속성 추론 위험을 증가시킬 수 있음(특정 조건에서 결과 확률이 과도하게 쏠릴 때).
- 동일지역 우선 재할당: 접근성 관점에서 타당하지만, 지역·병원 단위의 소규모 셀(특정 일자/병원/연령/성별)로 인한 희귀도 증가 가능.
- 품질 게이트: 통계·임상 일관성 검사는 프라이버시 보장을 제공하지 않음(식별/추론 위험과 독립).
- 캐싱/로깅: 분석 캐시(pickle)와 메타데이터(JSON), 파이프라인 로그가 소규모 빈도 정보를 외부로 노출할 잠재 위험.

### 정량적 위험 평가 지표(권장)
- k-익명성: 주요 준식별자 집합(QIs: `pat_age_gr, pat_sex, pat_do_cd, emorg_cd, vst_dt`) 기준 k ≥ 5/10 강제, k 미만 셀 비보존/병합.
- l-다양성/t-근접성: 민감속성(`ktas_fstu, emtrt_rust`)의 다양성과 원본 분포 근접성 상한 설정(희귀 카테고리에서 합성 분포 과적합 방지).
- ε-차등프라이버시 회계: 분포 추정(월/요일/시간/조건부) 시 DP 메커니즘(라플라스/가우시안 노이즈) 적용, 전체 ε, δ 회계 보고.
- 멤버십/속성 추론 AUC: 쉐도우 모델 기반 공격 시뮬레이션을 통한 공격자 성능(AUC, precision@k) 정기 측정 및 상한 규정.

### 권고 완화 전략(구현 우선순위)
1) 희귀 셀 억제·병합
   - 분포 캐싱 및 샘플링 시 최소 셀 카운트(예: m=10) 미만은 병합 또는 스무딩(디리클레/라플라스 보정). 현재 min_sample_size 수행 범위를 임상/시간/지역 모든 분포에 일관 적용.
   - 시간/병원/지역 단위의 소규모 조합은 상위 계층(대분류 지역, 전국+유형)으로 승격.

2) 노이즈 주입 및 랜다마이제이션
   - 시간 분포(일자/시간)와 조건부 분포(예: P(emtrt|KTAS, age))에 DP 노이즈 추가(ε 예산 지정) 후 정규화.
   - 재할당 결과(용량 후처리)에서 특정 날짜-병원-세그먼트의 잔여 희귀도를 완화하는 랜덤화(작은 비율의 균등 혼입).

3) 후처리 안전장치
   - k-익명성 검사: 합성 결과에 대해 QIs 기준 k<k_min 셀은 통폐합 또는 레코드 마스킹.
   - 빈도 상한: 보고/배포용 집계에서 최소 보고 임계치 적용(예: n<7 비표시 또는 [0,7) 구간).

4) 캐시·로그 통제
   - 캐시 파일 접근 통제(권한/암호화)와 보존기간 설정. 메타데이터에 원자료 통계(원시 샘플 문자열 등) 포함 금지.
   - 로그에서 드문 조합의 빈도 숫자 노출을 회피(상한/버킷팅).

5) 검증 파이프라인 강화
   - 기존 품질 게이트와 별도로 프라이버시 게이트 추가: k-익명성 통과율, MIA/AIA 공격 성능 상한 만족 시에만 배포.
   - 정기 리그레션: 분포 개선 변경이 프라이버시 지표를 악화시키지 않는지 자동 점검.

### 잔여 위험과 한계
- 목적-제한 합성: 의료 데이터 희귀 패턴 보존은 본질적으로 재식별 확률을 높일 수 있음. 정확도와 프라이버시는 상충 관계임.
- 외부 데이터와의 결합 위험은 제거 불가: 공개 통계, 보도자료, 병원 발표 자료 등과의 링크로 특정 이벤트가 드러날 수 있음.
- DP 미적용 상태: 현재 v0.3에는 공식 DP 보장이 포함되지 않음. 위 제언을 도입하기 전까지 공격자 모델에 대한 형식적 보장은 제공되지 않음.

### 실행 로드맵(요약)
- 단기(≤2주): 최소 셀 억제/병합 전면 적용, 캐시/로그 레드액션, 프라이버시 게이트(k-익명성) 추가.
- 중기(≤6주): 시간/조건부 분포에 DP 노이즈 주입, MIA/AIA 자동 평가 도구화, 보고용 최소 집계 임계치.
- 장기(≤3개월): 전 과정 DP 회계(ε, δ) 문서화, 조직 거버넌스/감사 절차 수립, 안전한 공유 프로토콜(액세스 제어, 계약) 정착.

## Overflow 용량 제약 상태와 한계 (v0.3)

본 버전의 병원 용량 제약(Overflow 재할당) 기능은 다음과 같은 제약으로 인해 완전하지 않습니다. 기본값은 비활성(disabled)이며, 명시적으로 활성화(--enable-overflow-redistribution)한 경우에도 일부 환경에서는 재할당이 발생하지 않을 수 있습니다.

### 관찰된 현상(예시 로그)

다음은 특정 실행에서 관측된 대표 로그이며, 다수의 Overflow 케이스가 탐지되었으나 재할당이 전혀 이루어지지 않았습니다.

```
INFO  Redistributing overflow patients
INFO  Found 4380 overflow cases
WARNING No available hospitals for redistribution (count: 1000)
WARNING No available hospitals for redistribution (count: 2000)
WARNING No available hospitals for redistribution (count: 3000)
WARNING No available hospitals for redistribution (count: 4000)
INFO  Successfully redistributed 0 patients
INFO  Validating capacity constraint results
WARNING Still 4380 capacity violations remain
WARNING Total excess patients: 282881
INFO  Capacity constraint processing completed
```

### 원인 분석
- 참조 데이터 부재: 병원 용량 메타(nedis_meta.hospital_capacity)가 없거나 불완전한 경우 동적 용량 한계를 계산할 수 없어, 동일 지역 내 여유 병원이 존재하지 않습니다.
- 재할당 정책 제약: 동일 지역 우선(same_region_first) + 지역 내 여유 용량이 0인 경우, 대안 탐색(이웃 지역/거리 기반)이 비활성 상태라 2차 후보가 없습니다.
- 보수적 한계치: 주말/공휴일 감산과 안전 여유(safety margin) 조합이 높은 과밀 환경에서 잔여 용량을 0에 가깝게 만듭니다.

### 현재 동작
- 기본 비활성: v0.3에서는 Overflow 재할당이 기본 비활성입니다. (--enable-overflow-redistribution 지정 시에만 시도)
- 비활성 시: 초기 배정 병원을 유지하고 overflow_flag=False, redistribution_method='disabled'로 결과를 기록합니다.

### 개선 계획
- 데이터 전제 강화: nedis_meta.hospital_capacity 스키마와 데이터 존재 검사 및 가이드 제공.
- 탐색 범위 확장: 동일 지역 실패 시, 인접 대분류/전국+유형 수준까지 계층적 대체 병원 탐색(거리 데이터 없이도 확률 기반 2nd/3rd choice 활용).
- 파라미터 노출: 주말/공휴일 계수, 안전 여유(safety margin)를 CLI/설정으로 더 세밀하게 제어.
- 성능 최적화: Overflow 케이스 집단 처리(batch) 및 로그 샘플링 고도화로 긴 실행 시에도 진전 상태 확인 가능.

> 결론적으로, v0.3의 Overflow 재할당은 “데이터 전제(용량 메타) 충족”과 “확장된 대체 정책” 없이는 기대 성능을 보장하지 못합니다. 운영 환경에서는 기본 비활성 상태로 사용하고, 필요 시 위 조건을 충족한 뒤 활성화할 것을 권장합니다.
