# NEDIS Synthetic Data Validation Suite

고급 통계적 검증, 패턴 분석, 시각화 기능을 갖춘 현대적인 검증 플랫폼입니다.

## 🎯 주요 기능

### 통계적 검증
- **연속형 변수**: Kolmogorov-Smirnov 검정, Wasserstein 거리
- **범주형 변수**: Chi-square 검정, Cramer's V 계수
- **상관관계 분석**: Pearson/Spearman 상관계수 검정
- **다변량 분석**: Multivariate normality test, PCA 기반 비교

### 패턴 분석
- **동적 패턴 발견**: Hierarchical fallback (소분류→대분류→전국)
- **시계열 패턴**: 시간 간격 분포, 순환 패턴 분석
- **임상 패턴**: 진단 코드 분포, 치료 패턴 분석
- **인구통계학적 패턴**: 연령/성별/지역 분포 분석

### 시각화 및 보고서
- **대화형 대시보드**: Dash 기반 웹 인터페이스 with Bootstrap
- **테이블별 컬럼 비교**: 즉시 비교, 통계 분석, 시각적 차이 강조
- **향상된 통계 표시**: 평균, 중위수, 표준편차, 왜도, 첨도, 분위수
- **분포 비교 차트**: 히스토그램 오버레이, 다중 테이블 지원
- **자동 보고서 생성**: PDF/Word/Excel/JSON 다중 형식 지원
- **CSV 내보내기**: 비교 결과 데이터 내보내기 기능
- **실시간 모니터링**: 검증 결과 실시간 스트리밍
- **맞춤형 알림**: 이상치 탐지 및 알림 시스템

### API 및 통합
- **REST API**: FastAPI 기반 RESTful API
- **GraphQL API**: 복잡한 쿼리 지원
- **웹소켓**: 실시간 데이터 스트리밍
- **Supabase 연동**: 클라우드 데이터베이스 지원
- **tRPC 연동**: 타입 안전한 API 호출

## 🏗️ 아키텍처

```
validator/
├── core/                    # 핵심 검증 엔진
│   ├── __init__.py
│   ├── validator.py         # 메인 검증 오케스트레이터
│   ├── config.py           # 설정 관리
│   └── database.py         # 데이터베이스 연결 관리
├── analysis/               # 분석 모듈
│   ├── __init__.py
│   ├── statistical.py      # 통계적 분석
│   ├── pattern.py          # 패턴 분석
│   ├── clinical.py         # 임상 데이터 분석
│   └── temporal.py         # 시계열 분석
├── visualization/          # 시각화 및 보고서
│   ├── __init__.py
│   ├── dashboard.py        # 웹 대시보드
│   ├── reports.py          # 보고서 생성
│   └── charts.py           # 차트 및 그래프
├── api/                    # API 레이어
│   ├── __init__.py
│   ├── routes.py           # API 엔드포인트
│   ├── schemas.py          # 데이터 스키마
│   └── middleware.py       # 미들웨어
├── utils/                  # 유틸리티
│   ├── __init__.py
│   ├── cache.py            # 캐싱 시스템
│   ├── logging.py          # 로깅 시스템
│   └── metrics.py          # 성능 메트릭스
└── cli.py                  # 명령줄 인터페이스
```

## 🚀 설치 및 사용

### 기본 설치

```bash
# 가상환경 생성 및 활성화
python -m venv validator_env
source validator_env/bin/activate  # Windows: validator_env\Scripts\activate

# 패키지 설치
pip install -r requirements.txt

# DuckDB 확장 프로그램 설치
duckdb -c "INSTALL 'httpfs'; LOAD 'httpfs';"
```

### Supabase 연동 (선택사항)

```bash
# Supabase CLI 설치
npm install -g supabase

# 프로젝트 초기화
supabase init

# 환경 변수 설정
export SUPABASE_URL="your_supabase_url"
export SUPABASE_ANON_KEY="your_anon_key"
```

### tRPC 백엔드 설정

```bash
# Node.js 프로젝트 생성 (선택사항)
npm init -y
npm install @trpc/server @trpc/client zod

# tRPC 서버 시작
npm run dev
```

## 📊 사용 예시

### 기본 검증 실행

```bash
# CLI를 통한 기본 검증
python -m validator.cli validate \
    --original-db nedis_original.duckdb \
    --synthetic-db nedis_synthetic.duckdb \
    --output-format html \
    --config validation_config.yaml
```

### API를 통한 검증

```python
from validator.api.client import ValidationClient

# 클라이언트 초기화
client = ValidationClient("http://localhost:8000")

# 비동기 검증 실행
result = await client.validate_async(
    original_db="nedis_original.duckdb",
    synthetic_db="nedis_synthetic.duckdb",
    validation_type="comprehensive"
)

# 결과 확인
print(f"Overall Score: {result['overall_score']}")
```

### 웹 대시보드 사용

```bash
# 대시보드 서버 시작 (방법 1)
python -m validator.visualization.dashboard

# 또는 테스트 스크립트 사용 (방법 2)
python test_dashboard.py

# 브라우저에서 접속
# http://localhost:8050
```

#### 🔬 테이블별 컬럼 비교 기능

1. **즉시 비교 실행**:
   - "Database Column Comparison" 섹션에서
   - Original DB와 Synthetic DB 경로 입력
   - "🔄 Compare Tables" 버튼 클릭하여 테이블 목록 로드

2. **테이블 선택 및 필터링**:
   - 드롭다운에서 비교할 테이블 선택 (다중 선택 가능)
   - 비교 타입 선택: All Columns / Numeric Only / Categorical Only

3. **결과 확인**:
   - 📊 **Numeric Columns**: 평균, 중위수, 표준편차, 분위수, 왜도, 첨도
   - 🏷️ **Categorical Columns**: 고유값 개수, 최빈값, 빈도
   - 📈 **Comparison Summary**: 전체 유사성 지수 및 평균 차이

4. **시각화 및 내보내기**:
   - "📈 Show Charts": 분포 히스토그램 비교 차트
   - "📊 Export to CSV": 비교 결과를 CSV 파일로 저장

#### 🎨 시각적 차이 강조

- **🟢 초록색**: 5% 이하 차이 (매우 유사)
- **🟡 노란색**: 5-15% 차이 (보통 차이)
- **🔴 빨간색**: 15% 이상 차이 (큰 차이)

### REST API 사용

```bash
# API 서버 시작
python -m validator.api.routes

# API 문서 확인
# http://localhost:8000/docs

# 검증 실행 예시
curl -X POST "http://localhost:8000/validate" \
     -H "Content-Type: application/json" \
     -d '{
       "original_db": "../nedis_data.duckdb",
       "synthetic_db": "../nedis_synth_2017.duckdb",
       "validation_type": "comprehensive",
       "sample_size": 50000
     }'
```

## 🔧 설정 파일

### validation_config.yaml

```yaml
# 검증 설정
validation:
  significance_level: 0.05
  sample_size: 50000
  enable_caching: true
  cache_ttl: 3600

# 통계적 검증 설정
statistics:
  ks_threshold: 0.05
  chi2_threshold: 0.05
  correlation_threshold: 0.1
  wasserstein_threshold: 0.1

# 패턴 분석 설정
patterns:
  min_sample_size: 10
  confidence_threshold: 0.95
  hierarchical_fallback: true
  time_gap_analysis: true

# 시각화 설정
visualization:
  enable_dashboard: true
  dashboard_port: 8050
  report_formats: ["html", "pdf", "json"]
  chart_theme: "default"

# API 설정
api:
  host: "0.0.0.0"
  port: 8000
  enable_cors: true
  rate_limit: 100

# 데이터베이스 설정
databases:
  original:
    path: "nedis_original.duckdb"
    schema: "nedis_original"
  synthetic:
    path: "nedis_synthetic.duckdb"
    schema: "nedis_synthetic"
```

## 📈 검증 메트릭스

### 종합 점수 계산

```
Overall Score = 0.4 × Statistical Score + 0.3 × Pattern Score + 0.2 × Clinical Score + 0.1 × Temporal Score
```

### 세부 점수

- **Statistical Score**: 통계적 유사성 (0-100)
- **Pattern Score**: 패턴 일치도 (0-100)
- **Clinical Score**: 임상 패턴 정확도 (0-100)
- **Temporal Score**: 시간 패턴 정확도 (0-100)

## 🔐 보안 및 프라이버시

- **차등 프라이버시**: 모든 분석에 차등 프라이버시 적용
- **데이터 익명화**: PII 정보 자동 마스킹
- **접근 제어**: 역할 기반 접근 제어 (RBAC)
- **감사 로깅**: 모든 검증 작업 로깅

## 📚 API 문서

자동 생성된 API 문서는 다음에서 확인 가능:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- GraphQL Playground: http://localhost:8000/graphql

## 🤝 기여 방법

1. Fork 프로젝트
2. 기능 브랜치 생성: `git checkout -b feature/amazing-feature`
3. 변경사항 커밋: `git commit -m 'Add amazing feature'`
4. 브랜치 푸시: `git push origin feature/amazing-feature`
5. Pull Request 생성

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 제공됩니다.

## 🆘 지원

문의사항은 다음 채널을 이용해 주세요:
- GitHub Issues: 버그 리포트 및 기능 요청
- 토론 포럼: 일반적인 질문 및 논의
- 이메일: 긴급 보안 이슈