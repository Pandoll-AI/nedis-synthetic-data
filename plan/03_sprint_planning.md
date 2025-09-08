# NEDIS 합성 데이터 생성 스프린트 계획

## 스프린트 개요

**총 기간**: 16주 (8 스프린트 × 2주)
**팀 구성**: 데이터 사이언티스트, 의료정보학 전문가, 백엔드 개발자, QA 엔지니어
**스프린트 목표**: 각 스프린트별 측정 가능한 성과물 달성

---

## Sprint 1: Foundation Setup (Week 1-2)

### 스프린트 목표
프로젝트 기반 인프라 구축 및 개발 환경 설정

### User Stories

#### Epic: Development Environment Setup
- **US-1.1**: 개발자로서 일관된 개발 환경을 위해 Docker 컨테이너 환경을 구축하고 싶다
  - **AC**: Docker Compose로 DuckDB, Python 환경 구성
  - **AC**: 모든 팀원이 동일한 환경에서 작업 가능
  - **Story Points**: 5

- **US-1.2**: 개발자로서 코드 품질 관리를 위해 CI/CD 파이프라인을 설정하고 싶다
  - **AC**: GitHub Actions로 자동 테스트 실행
  - **AC**: Code coverage 80% 이상 유지
  - **AC**: Pre-commit hooks 설정 (black, flake8, mypy)
  - **Story Points**: 8

#### Epic: Project Structure & Database Schema
- **US-1.3**: 개발자로서 명확한 프로젝트 구조를 통해 효율적으로 개발하고 싶다
  - **AC**: 개념서 기반 디렉토리 구조 생성
  - **AC**: 각 모듈별 `__init__.py` 파일 및 기본 클래스 정의
  - **Story Points**: 3

- **US-1.4**: 데이터 엔지니어로서 원본 데이터 로딩을 위한 DuckDB 스키마를 설정하고 싶다
  - **AC**: `nedis_original`, `nedis_meta`, `nedis_synthetic` 스키마 생성
  - **AC**: 모든 테이블 DDL 스크립트 작성 및 테스트
  - **AC**: 샘플 데이터로 스키마 검증
  - **Story Points**: 5

#### Epic: Basic Data Profiling
- **US-1.5**: 데이터 사이언티스트로서 원본 데이터의 특성을 파악하기 위해 기본 통계를 추출하고 싶다
  - **AC**: 전체 레코드 수, 결측값, 데이터 타입 검증
  - **AC**: 주요 변수별 분포 요약 통계 생성
  - **AC**: 데이터 품질 리포트 자동 생성
  - **Story Points**: 5

### Sprint 1 Definition of Done
- [ ] Docker 환경에서 DuckDB 연결 및 기본 쿼리 실행 가능
- [ ] CI/CD 파이프라인이 정상 작동하며 테스트 자동 실행
- [ ] 모든 데이터베이스 테이블이 생성되고 샘플 데이터 로딩 성공
- [ ] 기본 프로젝트 구조가 완성되고 각 모듈 import 가능
- [ ] 원본 데이터 기본 통계 리포트 생성

**Sprint Capacity**: 26 Story Points

---

## Sprint 2: Data Profiling & Metadata Extraction (Week 3-4)

### 스프린트 목표
원본 데이터로부터 메타데이터 추출 및 조건부 확률 테이블 생성

### User Stories

#### Epic: Population Profiling
- **US-2.1**: 데이터 사이언티스트로서 인구학적 패턴 분석을 위해 마진 테이블을 생성하고 싶다
  - **AC**: `PopulationProfiler` 클래스 구현
  - **AC**: 시도×연령×성별 조합별 연간 방문수 계산
  - **AC**: 계절별/요일별 가중치 계산 및 검증
  - **Story Points**: 8

- **US-2.2**: 의료정보학 전문가로서 병원별 용량 분석을 위해 통계 테이블을 생성하고 싶다
  - **AC**: `HospitalStatisticsExtractor` 클래스 구현
  - **AC**: 병원별 일평균 환자수, 표준편차 계산
  - **AC**: KTAS별 95th percentile 용량 계산
  - **Story Points**: 6

#### Epic: Conditional Probability Tables
- **US-2.3**: 임상 데이터 생성을 위해 KTAS 조건부 확률 테이블을 구축하고 싶다
  - **AC**: `ConditionalProbabilityExtractor` 클래스 구현
  - **AC**: 연령×성별×병원종별×내원수단 → KTAS 확률 계산
  - **AC**: 베이지안 평활화 (α=1.0) 적용
  - **Story Points**: 10

- **US-2.4**: 진단 코드 생성을 위해 진단별 조건부 확률 테이블을 구축하고 싶다
  - **AC**: ICD 진단 코드와 환자 특성 간 확률 관계 계산
  - **AC**: 주진단/부진단 구분 로직 구현
  - **AC**: 희귀 진단 코드 그룹화 (count < 10)
  - **Story Points**: 12

### Sprint 2 Definition of Done
- [ ] `nedis_meta.population_margins` 테이블이 완전히 채워짐
- [ ] `nedis_meta.hospital_capacity` 테이블에 모든 병원 통계 포함
- [ ] `nedis_meta.ktas_conditional_prob` 테이블 생성 및 검증
- [ ] `nedis_meta.diagnosis_conditional_prob` 테이블 생성 및 검증
- [ ] 모든 메타데이터 테이블에 대한 품질 검증 통과
- [ ] 단위 테스트 커버리지 85% 이상

**Sprint Capacity**: 36 Story Points

---

## Sprint 3: Population Volume Generation (Week 5-6)

### 스프린트 목표
Dirichlet-Multinomial 모델 기반 인구 볼륨 생성 시스템 구현

### User Stories

#### Epic: Volume Generation Engine
- **US-3.1**: 통계 모델링 전문가로서 Dirichlet-Multinomial 인구 생성기를 구현하고 싶다
  - **AC**: `PopulationVolumeGenerator` 클래스 구현
  - **AC**: 목표 총 방문수(920만)를 시도별로 분배
  - **AC**: Dirichlet 분포에서 확률벡터 샘플링
  - **AC**: Multinomial 분포로 연령×성별 조합 생성
  - **Story Points**: 13

- **US-3.2**: 시간 분석 전문가로서 NHPP 기반 일별 분해 시스템을 구현하고 싶다
  - **AC**: `NHPPTemporalGenerator` 클래스 구현
  - **AC**: 계절성 패턴 반영 (spring/summer/fall/winter)
  - **AC**: 요일 패턴 반영 (weekday/weekend)
  - **AC**: 공휴일 가중치 적용 (1.2x multiplier)
  - **Story Points**: 15

#### Epic: Quality Assurance for Volume Generation
- **US-3.3**: QA 엔지니어로서 생성된 볼륨의 통계적 타당성을 검증하고 싶다
  - **AC**: 원본 vs 합성 데이터 분포 비교 (KS test)
  - **AC**: 연간 총합 제약 조건 검증 (±1% 오차 범위)
  - **AC**: 계절성/요일 패턴 보존 검증
  - **Story Points**: 8

- **US-3.4**: 시스템 관리자로서 대용량 데이터 처리 성능을 최적화하고 싶다
  - **AC**: 메모리 효율적 배치 처리 구현
  - **AC**: 처리 진행률 실시간 모니터링
  - **AC**: 10만 레코드 기준 성능 벤치마크 수행
  - **Story Points**: 5

### Sprint 3 Definition of Done
- [ ] `nedis_synthetic.yearly_volumes` 테이블에 920만 레코드 분배 완료
- [ ] `nedis_synthetic.daily_volumes` 테이블에 365일 일별 분해 완료
- [ ] 원본 데이터와 통계적 유사성 검증 통과 (p-value > 0.05)
- [ ] 10만 레코드 처리 시간 < 10분
- [ ] 메모리 사용량 < 8GB peak

**Sprint Capacity**: 41 Story Points

---

## Sprint 4: Hospital Allocation System (Week 7-8)

### 스프린트 목표
중력모형 기반 병원 할당 및 용량 제약 처리 시스템 구현

### User Stories

#### Epic: Gravity Model Implementation
- **US-4.1**: 지리정보 분석 전문가로서 중력모형 기반 병원 선택 확률을 계산하고 싶다
  - **AC**: `HospitalGravityAllocator` 클래스 구현
  - **AC**: 시도-병원 간 거리 매트릭스 생성
  - **AC**: 병원 매력도 점수 계산 (용량×종별가중치)
  - **AC**: Huff 모델 확률 계산 (γ=1.5)
  - **Story Points**: 12

- **US-4.2**: 운영연구 전문가로서 용량 제약 하에서 환자 할당을 최적화하고 싶다
  - **AC**: 초기 확률적 할당 후 용량 초과 체크
  - **AC**: 오버플로우 재분배 로직 (권역→지역→지역기관)
  - **AC**: 동일 시도 내에서만 재분배 제약
  - **Story Points**: 15

#### Epic: IPF Marginal Adjustment
- **US-4.3**: 통계학자로서 IPF 알고리즘을 통해 마진 제약을 만족하고 싶다
  - **AC**: `IPFMarginalAdjuster` 클래스 구현
  - **AC**: 행/열 마진 조정을 반복 수행
  - **AC**: 수렴 조건 체크 (tolerance < 0.001)
  - **AC**: 최대 100회 반복 후 강제 종료
  - **Story Points**: 10

- **US-4.4**: 데이터 엔지니어로서 정수화 및 일관성 보장을 구현하고 싶다
  - **AC**: Controlled rounding (stochastic rounding)
  - **AC**: 최종 할당 결과의 정수 제약 만족
  - **AC**: 일별 총합 = 원본 일별 총합 보장
  - **Story Points**: 8

### Sprint 4 Definition of Done
- [ ] `nedis_meta.hospital_choice_prob` 테이블 생성 완료
- [ ] `nedis_synthetic.hospital_allocations` 테이블에 일별 할당 완료
- [ ] 모든 병원의 용량 제약 위반율 < 5%
- [ ] IPF 수렴 성공률 > 95%
- [ ] 10만 레코드 할당 처리 시간 < 15분

**Sprint Capacity**: 45 Story Points

---

## Sprint 5: Clinical Attribute Generation (Week 9-10)

### 스프린트 목표
DAG 기반 임상 속성 생성 및 의료 규칙 적용

### User Stories

#### Epic: DAG-based Clinical Generation
- **US-5.1**: 의료정보학 전문가로서 DAG 순서에 따라 임상 속성을 생성하고 싶다
  - **AC**: `ClinicalDAGGenerator` 클래스 구현
  - **AC**: vst_meth → ktas_fstu → emtrt_rust 순서 보장
  - **AC**: 각 단계별 조건부 확률 테이블 활용
  - **Story Points**: 13

- **US-5.2**: 응급의학 전문가로서 KTAS 등급별 현실적인 결과를 생성하고 싶다
  - **AC**: KTAS 1 → 중환자실입원 30%, 병실입원 50%, 사망 15%
  - **AC**: KTAS 5 → 귀가 95%, 자의퇴실 5%
  - **AC**: KTAS별 내원수단 분포 적용
  - **Story Points**: 10

#### Epic: Individual Record Generation
- **US-5.3**: 시스템 개발자로서 할당된 환자수만큼 개별 레코드를 생성하고 싶다
  - **AC**: 익명 환자ID 생성 (emorg_cd_patno_vst_dt_vst_tm)
  - **AC**: 방문시간 HHMM 랜덤 생성 (응급실 패턴 반영)
  - **AC**: 모든 필수 필드 생성 및 일관성 검증
  - **Story Points**: 12

- **US-5.4**: 의료 QA 전문가로서 생성된 임상 데이터의 의학적 타당성을 검증하고 싶다
  - **AC**: 연령-진단 불가능 조합 체크 (유아 심근경색 등)
  - **AC**: 성별-진단 불가능 조합 체크 (남성 임신 등)
  - **AC**: KTAS-결과 논리적 일관성 체크
  - **Story Points**: 8

### Sprint 5 Definition of Done
- [ ] `nedis_synthetic.clinical_records` 테이블에 기본 임상 속성 생성
- [ ] 모든 DAG 제약 조건 만족 (선후관계 위반 0%)
- [ ] 의학적 불가능 조합 발생률 < 0.1%
- [ ] 100만 레코드 생성 시간 < 2시간
- [ ] 임상 규칙 검증 통과율 > 99%

**Sprint Capacity**: 43 Story Points

---

## Sprint 6: Diagnosis & Temporal Features (Week 11-12)

### 스프린트 목표
진단 코드 생성 및 시간 변수(체류시간, 생체징후) 추가

### User Stories

#### Epic: Diagnosis Code Generation
- **US-6.1**: 임상코딩 전문가로서 ICD 기반 진단 코드를 생성하고 싶다
  - **AC**: `DiagnosisGenerator` 클래스 구현
  - **AC**: 환자 특성 기반 주진단 확률적 선택
  - **AC**: 주진단과 연관된 부진단 생성 (1-3개)
  - **Story Points**: 15

- **US-6.2**: 의료기록 관리자로서 입원 환자의 진단을 별도 생성하고 싶다
  - **AC**: emtrt_rust=31,32인 경우 입원 진단 생성
  - **AC**: ER 진단과 70% 일치, 30% 새로운 진단
  - **AC**: 평균 3-5개 진단 코드 생성
  - **Story Points**: 12

#### Epic: Duration and Vital Signs
- **US-6.3**: 응급의학 전문가로서 KTAS별 현실적인 체류시간을 생성하고 싶다
  - **AC**: `DurationGenerator` 클래스 구현
  - **AC**: KTAS별 로그정규분포 파라미터 적용
  - **AC**: 80% 정상 체류, 20% 장기 체류 혼합 모델
  - **Story Points**: 10

- **US-6.4**: 간호사로서 KTAS 중증도에 따른 생체징후를 생성하고 싶다
  - **AC**: `VitalSignsGenerator` 클래스 구현
  - **AC**: KTAS 1-2는 90%+ 측정, KTAS 5는 50% 측정
  - **AC**: 연령별 정상 범위 + KTAS별 이상 비율
  - **Story Points**: 13

### Sprint 6 Definition of Done
- [ ] `nedis_synthetic.diag_er` 테이블에 모든 ER 진단 생성
- [ ] `nedis_synthetic.diag_adm` 테이블에 입원 진단 생성
- [ ] 모든 clinical_records에 체류시간 (otrm_dt, otrm_tm) 추가
- [ ] 생체징후 필드 (vst_sbp, vst_dbp, vst_per_pu 등) 생성
- [ ] 진단 코드 유효성 검증 100% 통과
- [ ] 100만 레코드 처리 시간 < 3시간

**Sprint Capacity**: 50 Story Points

---

## Sprint 7: Validation Framework (Week 13-14)

### 스프린트 목표
종합적 품질 검증 시스템 구축 및 검증 수행

### User Stories

#### Epic: Statistical Validation
- **US-7.1**: 통계학자로서 원본과 합성 데이터의 분포 유사성을 검증하고 싶다
  - **AC**: `StatisticalValidator` 클래스 구현
  - **AC**: 연속형 변수 KS test (p-value > 0.05)
  - **AC**: 범주형 변수 Chi-square test
  - **AC**: 상관관계 매트릭스 차이 < 0.05
  - **Story Points**: 12

- **US-7.2**: 의료 QA 전문가로서 임상 규칙 준수를 자동 검증하고 싶다
  - **AC**: `ClinicalRuleValidator` 클래스 구현
  - **AC**: JSON 설정 파일 기반 규칙 엔진
  - **AC**: 연령-진단, 시간-일관성, KTAS-결과 규칙
  - **Story Points**: 15

#### Epic: Privacy Protection Validation
- **US-7.3**: 프라이버시 전문가로서 재식별 위험을 평가하고 싶다
  - **AC**: `PrivacyValidator` 클래스 구현
  - **AC**: Nearest neighbor distance 계산 (5th percentile)
  - **AC**: Membership inference attack 테스트
  - **Story Points**: 18

- **US-7.4**: 시스템 관리자로서 검증 결과를 자동 리포팅하고 싶다
  - **AC**: HTML 형태 품질 리포트 자동 생성
  - **AC**: 모든 검증 결과 통합 대시보드
  - **AC**: 임계값 위반 시 알림 시스템
  - **Story Points**: 8

### Sprint 7 Definition of Done
- [ ] 모든 통계적 검증 테스트 통과 (pass rate > 90%)
- [ ] 임상 규칙 위반율 < 1%
- [ ] 프라이버시 위험 평가 "Low Risk" 등급
- [ ] 자동화된 품질 리포트 생성 기능 완성
- [ ] 100만 레코드 검증 시간 < 30분

**Sprint Capacity**: 53 Story Points

---

## Sprint 8: Optimization & Final Integration (Week 15-16)

### 스프린트 목표
베이지안 최적화 및 전체 시스템 통합, 920만 레코드 생성

### User Stories

#### Epic: Reinforcement Learning Optimization
- **US-8.1**: 최적화 전문가로서 강화학습 기반 동적 가중치 최적화를 수행하고 싶다
  - **AC**: `NEDISWeightOptimizer` 클래스 구현 (PPO 알고리즘)
  - **AC**: 정책/가치 네트워크 구성 (PyTorch)
  - **AC**: 17개 시도별 계절/요일/중력모형 가중치 동적 조정
  - **AC**: 보상 함수 구현 (통계적 유사성, 임상적 타당성, 프라이버시, 생성시간)
  - **Story Points**: 25

- **US-8.2**: AI 훈련 전문가로서 강화학습 훈련 루프를 구현하고 싶다
  - **AC**: `RLTrainingLoop` 클래스 구현
  - **AC**: 에피소드별 소규모 데이터 생성 및 평가 (10K 샘플)
  - **AC**: PPO 손실함수 및 그래디언트 업데이트
  - **AC**: 최적 가중치 자동 저장 및 전체 재생성
  - **Story Points**: 20

#### Epic: Hybrid Optimization System
- **US-8.3**: 시스템 아키텍트로서 강화학습과 베이지안 최적화 하이브리드 시스템을 구축하고 싶다
  - **AC**: 강화학습 실패 시 베이지안 최적화 자동 백업
  - **AC**: 두 방법의 결과 비교 및 최적 선택
  - **AC**: 하이브리드 모드 설정 파라미터
  - **Story Points**: 12

- **US-8.4**: 시스템 관리자로서 강화학습 훈련 과정을 모니터링하고 싶다
  - **AC**: 실시간 훈련 대시보드 (에피소드별 보상, 손실)
  - **AC**: 가중치 변화 시각화
  - **AC**: 수렴 상태 자동 감지 및 알림
  - **Story Points**: 8

#### Epic: Full Scale Production Run
- **US-8.5**: 프로덕션 관리자로서 920만 레코드 전체 데이터셋을 생성하고 싶다
  - **AC**: 전체 파이프라인 24시간 내 완료
  - **AC**: 메모리 사용량 < 64GB peak
  - **AC**: 모든 품질 검증 자동 통과
  - **Story Points**: 15

- **US-8.6**: 사용자로서 최종 데이터셋과 상세한 품질 리포트를 받고 싶다
  - **AC**: Parquet 형식 최종 데이터셋 출력
  - **AC**: 종합 품질 리포트 (30+ 페이지) 생성
  - **AC**: 사용자 가이드 및 데이터 딕셔너리 제공
  - **Story Points**: 5

### Sprint 8 Definition of Done
- [ ] 강화학습 기반 가중치 최적화 시스템 구현 완료
- [ ] PPO 알고리즘 100 에피소드 훈련 성공적 수렴
- [ ] 하이브리드 최적화 시스템 (RL + 베이지안) 구현
- [ ] 920만 레코드 합성 데이터 생성 성공
- [ ] 최종 품질 점수 > 0.85 (4개 메트릭 가중 평균)
- [ ] RL 훈련 과정 모니터링 대시보드 구축
- [ ] 프로덕션 환경 배포 및 사용자 인수 테스트 완료
- [ ] 완전한 문서화 및 지식 이전 완료

**Sprint Capacity**: 70 Story Points

---

## 스프린트 관리 프로세스

### Daily Standups
- **시간**: 매일 오전 9:00 (15분)
- **형식**: 어제 한 일, 오늘 할 일, 블로커
- **추가 논의**: 데이터 품질 이슈, 성능 문제

### Sprint Review & Demo
- **대상**: 의료진 자문단, PO, 스테이크홀더
- **데모 내용**: 실제 합성 데이터 샘플 및 품질 메트릭
- **피드백 수집**: 의학적 타당성, 사용성 개선사항

### Sprint Retrospective
- **형식**: Start/Stop/Continue + Action Items
- **중점 영역**: 데이터 품질, 팀 협업, 기술적 도전과제
- **개선 실행**: 다음 스프린트에 반영

### 위험 관리 프로세스
- **주간 위험 검토**: 매주 금요일 30분
- **위험 카테고리**: 기술적, 의료적, 성능, 일정
- **에스컬레이션**: 높은 위험 즉시 스테이크홀더 보고

이 스프린트 계획은 복잡한 의료 데이터 합성 생성 프로젝트를 관리 가능한 단위로 분해하고, 각 단계별로 명확한 성과 측정과 품질 보증을 제공하도록 설계되었습니다.