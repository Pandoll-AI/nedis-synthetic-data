# NEDIS 합성 데이터 독립형 대시보드

## 개요
이 파일은 서버 종속성 없이 작동하는 완전 독립형 NEDIS 합성 데이터 분석 대시보드입니다.

## 생성 정보
- 생성 시간: 2025-09-09 08:20:49
- 파일: nedis_standalone_dashboard.html
- 크기: 2.4 MB

## 사용 방법

### 1. 로컬 실행
```bash
# 방법 1: 브라우저에서 직접 열기
# 파일 탐색기에서 nedis_standalone_dashboard.html 더블클릭

# 방법 2: 명령줄에서 열기 (macOS)
open nedis_standalone_dashboard.html

# 방법 3: 명령줄에서 열기 (Windows)
start nedis_standalone_dashboard.html

# 방법 4: 명령줄에서 열기 (Linux)
xdg-open nedis_standalone_dashboard.html
```

### 2. 웹 서버에 배포
```bash
# 정적 파일 서버에 업로드
# - GitHub Pages
# - Netlify
# - Vercel
# - AWS S3 Static Website
# - 기타 정적 호스팅 서비스
```

## 기능

### 주요 차트
- 연령별 분포 비교
- 성별 분포 비교
- 지역별 분포 비교
- KTAS 중증도 분포 비교
- 병원별 분포 비교
- 월별 패턴 비교

### 상호작용 기능
- 차트 타입 변경 (막대, 선형, 원형, 도넛)
- 데이터 표시 모드 (비율, 절대값, 정규화)
- 비교 모드 선택
- 품질 평가 지표

### 반응형 디자인
- 모바일 및 태블릿 지원
- 다양한 화면 크기 최적화

## 기술 스택
- HTML5
- CSS3 (Bootstrap 5.3)
- JavaScript (ES6+)
- Chart.js 4.3
- Font Awesome 6.0

## 브라우저 호환성
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 특징
- 서버 불필요 (완전 클라이언트 사이드)
- 외부 API 호출 없음
- 모든 데이터 임베딩
- 오프라인 작동 가능

## 데이터 보안
- 모든 데이터가 클라이언트 사이드에서 처리됨
- 외부 서버로 데이터 전송 없음
- HTTPS 연결 권장 (배포 시)

## 성능 최적화
- 데이터 압축 및 최적화
- 지연 로딩 (필요시)
- 차트 렌더링 최적화

## 문제 해결

### 차트가 표시되지 않는 경우
1. 브라우저의 JavaScript가 활성화되어 있는지 확인
2. 브라우저 개발자 도구(F12)에서 오류 메시지 확인
3. 인터넷 연결 상태 확인 (CDN 리소스 로딩용)

### 데이터가 로딩되지 않는 경우
1. 파일이 완전히 다운로드되었는지 확인
2. 파일이 손상되지 않았는지 확인
3. 브라우저 캐시 삭제 후 재시도

## 업데이트
새로운 데이터로 대시보드를 업데이트하려면:
1. 새로운 데이터로 `create_embedded_dashboard.py` 스크립트 실행
2. 생성된 새 HTML 파일로 교체

## 라이선스
이 대시보드는 NEDIS 합성 데이터 프로젝트의 일부입니다.

---
생성 시간: 2025-09-09 08:20:49
