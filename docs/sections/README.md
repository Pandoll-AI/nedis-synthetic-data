# NEDIS Documentation - Sections

`docs/sections/` 디렉토리에 포함된 기술 문서 목록입니다.

## Available Sections

| 파일 | 내용 | 참조 소스 |
|------|------|----------|
| [03_pattern_analysis_system.md](03_pattern_analysis_system.md) | SQL 기반 동적 패턴 분석, 계층적 fallback, 캐싱 | `src/analysis/pattern_analyzer.py` |
| [04_synthetic_data_generation.md](04_synthetic_data_generation.md) | 벡터화 환자 생성 4단계, 시간 할당, 용량 제약 | `src/vectorized/*.py` |
| [06_privacy_enhancement_framework.md](06_privacy_enhancement_framework.md) | K-anonymity, differential privacy, generalization | `src/privacy/*.py` |

## Key Principles

- **Dynamic Pattern Learning**: 하드코딩 없이 원본 데이터에서 모든 분포 학습
- **Hierarchical Fallback**: 데이터 희소 시 상위 범주로 자동 대체
- **Vectorized Operations**: NumPy 기반 벡터화로 약 50배 성능 향상
- **Privacy-First**: K-anonymity, L-diversity, differential privacy 구현 완료

상위 인덱스는 [NEDIS_DOCUMENTATION_INDEX.md](../NEDIS_DOCUMENTATION_INDEX.md)를 참조하세요.
