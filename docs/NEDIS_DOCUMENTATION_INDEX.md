# NEDIS Synthetic Data Generation System - Documentation Index

## Overview

NEDIS 합성 데이터 생성 시스템의 기술 문서 인덱스입니다. 모든 문서는 실제 구현 코드를 기반으로 작성되었습니다.

---

## Core Technical Documents

### Section Files (`docs/sections/`)

| 문서 | 설명 | 참조 소스 |
|------|------|----------|
| [Pattern Analysis System](sections/03_pattern_analysis_system.md) | SQL 기반 경험적 분포 추출, 4단계 계층적 fallback, 10가지 패턴 분석, 캐싱 전략 | `src/analysis/pattern_analyzer.py` |
| [Synthetic Data Generation](sections/04_synthetic_data_generation.md) | 4단계 벡터화 환자 생성, 조건부 시간 분포 블렌딩, 용량 제약 후처리 | `src/vectorized/*.py` |
| [Privacy Enhancement Framework](sections/06_privacy_enhancement_framework.md) | K-anonymity, L-diversity, differential privacy, generalization | `src/privacy/*.py` |

### Standalone Documents

| 문서 | 설명 |
|------|------|
| [Core Algorithm and Workflows](core_algorithm_and_workflows.md) | 전체 파이프라인 원칙, 아키텍처, 워크플로우 요약 |
| [Vectorized Generation Algorithm](vectorized_generation_algorithm.md) | 벡터화 생성 알고리즘 상세 설명 |
| [Comprehensive System Analysis](comprehensive_system_analysis.md) | 시스템 아키텍처, 성능 분석, 프라이버시 위험 평가 |
| [Privacy Enhancement Implementation](privacy_enhancement_implementation.md) | 프라이버시 모듈 구현 상세 |
| [Time Gap Synthesis Implementation](time_gap_synthesis_implementation.md) | 시간 차이(발병-내원, 내원-퇴실 등) 합성 설계 |

### Revision History

| 문서 | 내용 |
|------|------|
| [Revision v0.5](revision_v0.5.md) | NEDIS 4.0 마이그레이션 + 임상 일관성 (결측값 보존, 조건부 생성, KTAS 코드) |
| [Revision v0.4](revision_v0.4.md) | 시간 패턴 버그 수정 |
| [Revision v0.3](revision_v0.3.md) | 동적 분포 전환, 품질 게이트 |

---

## Key Design Principles

### Dynamic Pattern Learning
- 모든 분포는 원본 데이터에서 SQL 윈도우 함수로 학습
- 하드코딩된 확률이나 가중치 없음
- Pickle/JSON 캐싱으로 반복 분석 방지

### Hierarchical Fallback
- 데이터 희소 시 상위 범주로 자동 대체
- KTAS: region+hospital_type → major_region → national → overall
- 최소 샘플 크기(10) 검증

### Performance
- 벡터화: ~50배 성능 향상 (322K 레코드 약 7초)
- NumPy 배열 연산 + 그룹별 일괄 처리
- 시간 분리 전략: 속성 생성과 날짜 할당 분리

### Privacy Protection
- K-anonymity, L-diversity, T-closeness 검증
- Differential privacy (Laplace/Gaussian) 적용 가능
- Age/region/temporal generalization
- `EnhancedSyntheticGenerator` 7단계 파이프라인

### Browser-Based Generation
- 단일 HTML 파일 기반 브라우저 생성기 (`outputs/html_generator/nedis_generator.html`)
- 패턴을 zlib+base64로 HTML에 임베딩
- Multi-worker 병렬 생성 지원

---

## Related Project Files

| 파일 | 용도 |
|------|------|
| [CLAUDE.md](../CLAUDE.md) | 개발 가이드라인 및 코딩 규칙 |
| [CHANGELOG.md](../CHANGELOG.md) | 전체 버전별 변경 이력 |
| [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) | 초기 구현 계획 (Historical) |
| [NEDIS_CODEBOOK.md](../NEDIS_CODEBOOK.md) | 2017 표본자료 코드북 |
| `reference/NEDIS_VARIABLE_MAPPING.md` | 표본자료 ↔ NEDIS 4.0 변수 매칭표 |

---

*최종 갱신: 2026-03-07. 실제 소스 코드 기반으로 검증된 문서입니다.*
