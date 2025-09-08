# NEDIS 합성 데이터 생성 개발 로드맵

## 프로젝트 개요

**목표**: 응급의료정보시스템(NEDIS) 2017년 데이터(920만 레코드)의 통계적 특성을 보존하면서 프라이버시를 보호하는 합성 데이터 생성 시스템 구축

**기간**: 16주 (4개월)
**팀 규모**: 3-4명 (데이터 사이언티스트, 의료정보학 전문가, 백엔드 개발자, QA)

## 기술 스택

### 핵심 기술
- **데이터베이스**: DuckDB (고성능 분석 엔진)
- **개발 언어**: Python 3.9+
- **통계 라이브러리**: SciPy, NumPy, scikit-learn
- **최적화**: scikit-optimize (베이지안 최적화), PyTorch (강화학습 PPO)
- **설정 관리**: YAML, JSON
- **테스팅**: pytest, hypothesis

### 보조 도구
- **데이터 검증**: pandas, great_expectations
- **시각화**: matplotlib, seaborn, plotly
- **모니터링**: logging, tqdm
- **문서화**: Sphinx, mkdocs

## 개발 Phase 및 일정

### Phase 1: Foundation (Week 1-4)
**목표**: 기반 인프라 구축 및 데이터 프로파일링

#### Sprint 1 (Week 1-2)
- [ ] 프로젝트 환경 설정 및 디렉토리 구조 생성
- [ ] DuckDB 스키마 설계 및 구현
- [ ] 원본 데이터 로딩 및 기본 통계 추출
- [ ] 개발 환경 및 CI/CD 파이프라인 구축

#### Sprint 2 (Week 3-4)
- [ ] 인구학적 마진 추출 (`PopulationProfiler`)
- [ ] 병원별 용량 통계 계산 (`HospitalStatisticsExtractor`)
- [ ] 조건부 확률 테이블 생성 (`ConditionalProbabilityExtractor`)
- [ ] 첫 번째 품질 검증 프레임워크 구현

**Deliverables**:
- 완전한 메타데이터 테이블 (`nedis_meta` 스키마)
- 기본 데이터 품질 리포트
- 단위 테스트 커버리지 80%+

### Phase 2: Core Pipeline (Week 5-8)
**목표**: 핵심 합성 데이터 생성 파이프라인 구현

#### Sprint 3 (Week 5-6)
- [ ] Dirichlet-Multinomial 인구 생성기 (`PopulationVolumeGenerator`)
- [ ] 비균질 포아송 과정 시간 분해 (`NHPPTemporalGenerator`)
- [ ] 연간 → 일별 볼륨 변환 로직 구현
- [ ] 계절성 및 요일 패턴 반영

#### Sprint 4 (Week 7-8)
- [ ] 중력모형 병원 할당기 (`HospitalGravityAllocator`)
- [ ] 용량 제약 처리 및 오버플로우 재분배
- [ ] IPF 마진 조정기 (`IPFMarginalAdjuster`)
- [ ] 중간 규모 테스트 (10만 레코드)

**Deliverables**:
- 일별 병원 할당 시스템
- 용량 제약 하에서 현실적 분포
- 성능 벤치마크 (처리 속도, 메모리 사용량)

### Phase 3: Clinical Features (Week 9-12)
**목표**: 의료 도메인 특화 속성 생성

#### Sprint 5 (Week 9-10)
- [ ] DAG 기반 임상 속성 생성기 (`ClinicalDAGGenerator`)
- [ ] KTAS 등급 및 내원수단 생성
- [ ] 주증상 및 응급치료결과 생성
- [ ] 의료진 검토를 통한 임상 규칙 검증

#### Sprint 6 (Week 11-12)
- [ ] ICD 기반 진단 코드 생성기 (`DiagnosisGenerator`)
- [ ] 주진단/부진단 논리 구현
- [ ] 체류시간 및 입원 기간 모델링 (`DurationGenerator`)
- [ ] 생체징후 생성기 (`VitalSignsGenerator`)

**Deliverables**:
- 완전한 임상 레코드 생성 시스템
- 의료 도메인 검증 통과
- 중간 규모 합성 데이터셋 (100만 레코드)

### Phase 4: Validation & Optimization (Week 13-16)
**목표**: 품질 보증 및 성능 최적화

#### Sprint 7 (Week 13-14)
- [ ] 통계적 검증 시스템 (`StatisticalValidator`)
- [ ] 임상 규칙 검증기 (`ClinicalRuleValidator`)
- [ ] 프라이버시 검증 (`PrivacyValidator`)
- [ ] Nearest neighbor distance 및 membership inference 테스트

#### Sprint 8 (Week 15-16)
- [ ] 강화학습 기반 동적 가중치 최적화 (`RLWeightOptimizer`)
- [ ] PPO 알고리즘 훈련 루프 구현 (`RLTrainingLoop`)
- [ ] 베이지안 최적화와 강화학습 하이브리드 시스템
- [ ] 다목적 최적화 (fidelity, utility, privacy, clinical validity)
- [ ] 전체 시스템 통합 및 920만 레코드 생성 테스트
- [ ] 최종 품질 리포트 및 문서화

**Deliverables**:
- 프로덕션 준비된 합성 데이터 생성 시스템
- 완전한 검증 리포트 (통계적, 임상적, 프라이버시)
- 사용자 가이드 및 API 문서

## 마일스톤 및 성공 지표

### 마일스톤
1. **Week 4**: 메타데이터 추출 완료
2. **Week 8**: 10만 레코드 합성 데이터 생성 성공
3. **Week 12**: 100만 레코드 임상 검증 통과
4. **Week 16**: 920만 레코드 최종 데이터셋 완성

### 성공 지표
- **통계적 유사성**: KS test p-value > 0.05 (주요 변수 90%+)
- **임상적 타당성**: Rule violation rate < 1%
- **프라이버시 보호**: 5th percentile NN distance > threshold
- **유틸리티**: TSTR AUC > 0.85 (입원 예측 태스크)
- **성능**: 920만 레코드 생성 < 24시간
- **RL 최적화**: 강화학습 수렴 시 품질 점수 > 0.85

## 위험 요소 및 대응책

### 높은 위험
- **의료 도메인 복잡성**: 의료진 자문단 구성, 문헌 조사 강화
- **대용량 데이터 처리**: 분산 처리, 배치 최적화, 메모리 관리
- **프라이버시 위험**: K-익명성, 차분 프라이버시 기법 적용

### 중간 위험
- **통계 모델 정확성**: 시뮬레이션 검증, A/B 테스트
- **성능 최적화**: 프로파일링, 병렬 처리 최적화
- **강화학습 수렴 불안정성**: 하이퍼파라미터 튜닝, 베이지안 백업 옵션

### 낮은 위험
- **개발 일정 지연**: 버퍼 시간 확보, 우선순위 조정

## 품질 보증 전략

### 테스트 전략
- **단위 테스트**: 각 모듈별 90% 커버리지 목표
- **통합 테스트**: Phase별 엔드투엔드 테스트
- **성능 테스트**: 규모별 벤치마크 (1K/10K/100K/1M/10M)
- **검증 테스트**: 통계적/임상적 타당성 자동 검증

### 모니터링
- **실시간 모니터링**: 생성 과정 진행률, 오류율, 성능 지표
- **품질 대시보드**: 분포 비교, 규칙 위반 현황
- **알림 시스템**: 임계값 초과 시 자동 알림

## 리소스 요구사항

### 하드웨어
- **개발 환경**: 32GB RAM, 8-core CPU, 1TB SSD
- **테스트 환경**: 64GB RAM, 16-core CPU, 2TB SSD
- **프로덕션**: 128GB RAM, 32-core CPU, 10TB 스토리지

### 인력
- **데이터 사이언티스트** (1명): 통계 모델링, 알고리즘 구현
- **의료정보학 전문가** (1명): 임상 규칙, 도메인 검증
- **백엔드 개발자** (1명): 파이프라인 구축, 성능 최적화
- **QA 엔지니어** (1명): 테스트 자동화, 품질 검증

## 배포 및 운영

### 배포 전략
- **Docker 컨테이너**: 환경 표준화 및 배포 자동화
- **파이프라인 오케스트레이션**: Apache Airflow 또는 Prefect
- **버전 관리**: Git-based 코드 버전 + 데이터셋 버전 추적

### 운영 모니터링
- **로그 관리**: 구조화된 로깅, 중앙화된 로그 수집
- **메트릭 수집**: 생성 품질, 성능, 시스템 리소스
- **알람 설정**: 품질 임계값 이하, 시스템 오류 발생 시

이 로드맵은 점진적이고 위험 관리 중심의 접근법으로 설계되었으며, 각 Phase별로 명확한 성과물과 검증 기준을 제시하여 프로젝트 성공률을 극대화하고자 합니다.