# Changelog

NEDIS 합성 데이터 생성 시스템의 전체 변경 이력입니다.

---

## [v0.5] - 2026-03-07

### NEDIS 4.0 변수명 전면 마이그레이션
- 39개 Python 파일, ~2,097개 참조를 표본자료 변수명(`emorg_cd`, `vst_dt` 등)에서 NEDIS 4.0 DB 칼럼 ID(`ptmiemcd`, `ptmiindt` 등)로 전환
- `src/core/nedis4_converter.py` 신규 모듈: 26쌍 변수 매핑 단일 진실 소스
- DuckDB VIEW 자동 생성(`_ensure_nedis4_view`): 소스 DB가 구 변수명 사용 시 런타임 alias
- 값 변환: 성별 M/F→1/2 (VIEW), 연령군→합성 YYYYMMDD, 시도코드→합성 12자리 우편번호

### 임상 일관성 개선 (Phase A/B/C)

#### Phase A: 구조적 결측값 보존
- `analyze_missing_value_rates()`: 독립/상관/조건부 결측률 분석
- `_apply_missing_values()`: KTAS·주증상·진료과 co-missing 구조 재현 (±0.6% 이내)
- 패턴 분석 SQL에서 `'-'` 결측값 제외 (`AND col != '-'`)

#### Phase B: 조건부 생성 체인
- KTAS → 주증상 P(sym|ktas,age,sex) → 진료과 P(dept|ktas,sym) → 치료결과
- 계층적 대안: ktas_age_sex → ktas_age → ktas → age_sex (주증상), ktas_sym → ktas → sym → all (진료과)
- Categorical score: 0.697 → 0.963 (+38%)

#### Phase C: KTAS 프로토콜 코드 합성
- `ptmikpr1` 6자리 코드 합성: 연령구분(A/P) + 대분류 + 중분류 + 소분류 + 감염코드
- 주증상(UMLS) → 신체계통 대분류 매핑 (`_SYMPTOM_MAJOR_CLASS`)
- DDL 변경: `ptmikpr1 INTEGER` → `VARCHAR`

### 300K 검증 결과
| 항목 | 지표 |
|------|------|
| KTAS 등급 | JSD 0.000015 |
| 성별 / 연령 / 내원수단 | JSD < 0.000005 |
| 결측률 (7개 변수) | 전 항목 ±0.6% |
| KTAS-주증상 Top-3 일치 | 14/15 |
| KTAS-진료과 Top-3 일치 | 14/15 |
| 활력징후 평균 차이 | ≤ 0.2 |

---

## [v0.4] - 2026-03-02

### 시간 패턴 버그 수정
- 조건부 시간 분포 평탄화 버그 수정 (`count` → `sum` 집계)
- DOW 매핑 불일치 수정 (Python `weekday()` 0-based ↔ SQL `dayofweek` 1-based)
- 시간별 분포 정규화 누락 수정

---

## [v0.3] - 2026-02-28

### 동적 분포 및 품질 게이트
- 동적 분포 기반 생성으로 전면 전환: 하드코딩 분포 완전 제거
- 배치 KTAS 생성: 계층적 대안 4단계 (`region+hospital_type` → `major_region` → `national` → `overall`)
- Same-region-first overflow 전략
- 품질 게이트: JSD·KL 기반 자동 검증
- CLI 기본값 설정 및 fallback 제거

---

## [v0.2] - 2026-02-25

### 다변량 상관 및 시간 흐름
- 다변량 상관관계 보존 (`feat: Enhance synthetic data realism`)
- 시간 흐름 합성: 발병→내원→퇴실→입원 시간 차이 패턴 학습
- 종합 시간 간격 합성기 (`ComprehensiveTimeGapSynthesizer`)

### 프라이버시 강화
- K-anonymity, L-diversity, T-closeness 검증 프레임워크
- Differential privacy (Laplace/Gaussian) 적용
- 연령/지역/시간 일반화

### 검증 시스템 모듈화
- `src/validation/` 모듈 분리: clinical, statistical, correlation, pipeline quality

---

## [v0.1] - 2026-02-20

### 초기 구현
- 벡터화 NEDIS 합성 데이터 생성 파이프라인
- NumPy 기반 배치 생성 (~50배 성능 향상)
- SQL 윈도우 함수 기반 패턴 분석 (`PatternAnalyzer`)
- DuckDB 기반 데이터 관리
- 브라우저 기반 HTML 생성기 (단일 파일, zlib+base64 패턴 임베딩)
- Multi-worker 병렬 생성 지원
- 반복 품질 루프 (`iterative_synthetic_quality_loop.py`)

---

*전체 revision 상세 문서: `docs/revision_v0.3.md`, `docs/revision_v0.4.md`, `docs/revision_v0.5.md`*
