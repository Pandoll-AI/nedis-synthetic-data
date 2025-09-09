#!/usr/bin/env python3
"""
NEDIS 합성 데이터 대시보드 - 독립형 HTML 생성기

이 스크립트는 서버리스 대시보드 데이터를 HTML 파일에 직접 임베딩하여
서버 종속성 없이 작동하는 독립형 대시보드를 생성합니다.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

def load_dashboard_data(data_file_path):
    """대시보드 데이터 로드 및 검증"""
    try:
        print(f"데이터 파일 로딩 중: {data_file_path}")
        
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_file_path}")
        
        file_size = os.path.getsize(data_file_path)
        print(f"파일 크기: {file_size / (1024*1024):.1f} MB")
        
        with open(data_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 데이터 구조 검증
        required_sections = ['metadata', 'original_data', 'synthetic_data']
        for section in required_sections:
            if section not in data:
                raise ValueError(f"필수 데이터 섹션이 없습니다: {section}")
        
        print("데이터 로딩 및 검증 완료")
        print(f"- 원본 레코드: {data['metadata'].get('original_records', 0):,}")
        print(f"- 합성 레코드: {data['metadata'].get('synthetic_records', 0):,}")
        
        return data
        
    except Exception as e:
        print(f"데이터 로딩 실패: {e}")
        sys.exit(1)

def optimize_data_for_embedding(data):
    """임베딩을 위한 데이터 최적화"""
    print("데이터 최적화 중...")
    
    # 원본 데이터 크기 계산
    original_size = len(json.dumps(data, ensure_ascii=False))
    print(f"원본 데이터 크기: {original_size / 1024:.1f} KB")
    
    # 불필요한 필드 제거 및 정밀도 조정
    optimized_data = {
        'metadata': data['metadata'],
        'original_data': {},
        'synthetic_data': {}
    }
    
    # 각 분포 데이터 최적화
    for data_type in ['original_data', 'synthetic_data']:
        if data_type in data:
            optimized_data[data_type] = optimize_distributions(data[data_type])
    
    # 비교 데이터가 있다면 추가
    if 'comparison' in data:
        optimized_data['comparison'] = data['comparison']
    
    # 최적화된 데이터 크기 계산
    optimized_size = len(json.dumps(optimized_data, ensure_ascii=False))
    reduction = ((original_size - optimized_size) / original_size) * 100
    
    print(f"최적화된 데이터 크기: {optimized_size / 1024:.1f} KB")
    print(f"크기 감소율: {reduction:.1f}%")
    
    return optimized_data

def optimize_distributions(data_section):
    """분포 데이터 최적화"""
    optimized_section = {}
    
    for key, distribution in data_section.items():
        if isinstance(distribution, list):
            optimized_distribution = []
            for item in distribution:
                optimized_item = {}
                for field, value in item.items():
                    if isinstance(value, float):
                        # 소수점 4자리로 제한
                        optimized_item[field] = round(value, 4)
                    else:
                        optimized_item[field] = value
                optimized_distribution.append(optimized_item)
            
            # 상위 N개 항목만 유지 (필요시)
            if len(optimized_distribution) > 100 and key not in ['age_distribution', 'sex_distribution', 'ktas_distribution']:
                optimized_distribution = optimized_distribution[:50]
            
            optimized_section[key] = optimized_distribution
        else:
            optimized_section[key] = distribution
    
    return optimized_section

def create_embedded_dashboard(template_file, data, output_file):
    """데이터를 임베딩한 독립형 대시보드 생성"""
    try:
        print(f"템플릿 파일 로딩: {template_file}")
        
        with open(template_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # 데이터를 JavaScript 변수로 변환
        data_js = f"""
        // 임베디드 대시보드 데이터
        // 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        // 원본 레코드: {data['metadata'].get('original_records', 0):,}
        // 합성 레코드: {data['metadata'].get('synthetic_records', 0):,}
        
        const EMBEDDED_DATA = {json.dumps(data, ensure_ascii=False, indent=2)};
        
        console.log('임베디드 데이터 로딩 완료:', EMBEDDED_DATA.metadata);
        """
        
        # HTML 파일에서 데이터 임베딩 지점 찾기 및 교체
        data_placeholder = 'const EMBEDDED_DATA = null; // 이 부분에 실제 데이터가 삽입됩니다'
        
        if data_placeholder in html_content:
            html_content = html_content.replace(data_placeholder, f'const EMBEDDED_DATA = {json.dumps(data, ensure_ascii=False)};')
            print("데이터 임베딩 완료 (인라인 방식)")
        else:
            # 대안: script 태그 교체 방식
            script_placeholder = '<!-- 데이터 삽입 지점 - 실제 구현 시 이 부분에 JSON 데이터가 삽입됩니다 -->\n    <script id="embedded-data">\n        // 이 부분에 실제 데이터가 삽입될 예정입니다.\n        // 현재는 외부 파일에서 로드하도록 설정되어 있습니다.\n    </script>'
            
            if script_placeholder in html_content:
                html_content = html_content.replace(script_placeholder, f'<!-- 임베디드 데이터 -->\n    <script id="embedded-data">\n{data_js}    </script>')
                print("데이터 임베딩 완료 (스크립트 태그 방식)")
            else:
                print("경고: 데이터 임베딩 지점을 찾을 수 없음. 기본 위치에 삽입합니다.")
                # </body> 태그 앞에 데이터 삽입
                html_content = html_content.replace('</body>', f'    <script>{data_js}    </script>\n</body>')
        
        # 메타데이터 업데이트
        generation_info = f"""
        <!-- 
        독립형 NEDIS 합성 데이터 대시보드
        생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        데이터 크기: {len(json.dumps(data, ensure_ascii=False)) / 1024:.1f} KB
        원본 레코드: {data['metadata'].get('original_records', 0):,}
        합성 레코드: {data['metadata'].get('synthetic_records', 0):,}
        서버 종속성: 없음 (완전 독립형)
        -->
        """
        
        # HTML head에 메타데이터 추가
        html_content = html_content.replace('<head>', f'<head>\n{generation_info}')
        
        # 파일 저장
        print(f"독립형 대시보드 저장 중: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        output_size = os.path.getsize(output_file)
        print(f"독립형 대시보드 생성 완료")
        print(f"- 파일 크기: {output_size / (1024*1024):.1f} MB")
        print(f"- 경로: {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"대시보드 생성 실패: {e}")
        sys.exit(1)

def validate_html_file(html_file):
    """생성된 HTML 파일 검증"""
    try:
        print(f"HTML 파일 검증 중: {html_file}")
        
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 필수 요소들 확인
        checks = {
            'HTML 기본 구조': '<html' in content and '</html>' in content,
            '데이터 임베딩': 'EMBEDDED_DATA' in content,
            'Chart.js 라이브러리': 'chart.js' in content or 'chart.umd.js' in content,
            'Bootstrap CSS': 'bootstrap' in content,
            '메인 컨테이너': 'main-container' in content,
            '차트 캔버스': '<canvas' in content,
        }
        
        all_passed = True
        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("HTML 파일 검증 완료: 모든 검사 통과")
        else:
            print("경고: 일부 검사에서 문제가 발견되었습니다.")
        
        return all_passed
        
    except Exception as e:
        print(f"HTML 파일 검증 실패: {e}")
        return False

def create_demo_instructions(html_file):
    """사용 안내서 생성"""
    instructions_file = str(html_file).replace('.html', '_README.md')
    
    content = f"""# NEDIS 합성 데이터 독립형 대시보드

## 개요
이 파일은 서버 종속성 없이 작동하는 완전 독립형 NEDIS 합성 데이터 분석 대시보드입니다.

## 생성 정보
- 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 파일: {os.path.basename(html_file)}
- 크기: {os.path.getsize(html_file) / (1024*1024):.1f} MB

## 사용 방법

### 1. 로컬 실행
```bash
# 방법 1: 브라우저에서 직접 열기
# 파일 탐색기에서 {os.path.basename(html_file)} 더블클릭

# 방법 2: 명령줄에서 열기 (macOS)
open {os.path.basename(html_file)}

# 방법 3: 명령줄에서 열기 (Windows)
start {os.path.basename(html_file)}

# 방법 4: 명령줄에서 열기 (Linux)
xdg-open {os.path.basename(html_file)}
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
생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(instructions_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"사용 안내서 생성: {instructions_file}")
    return instructions_file

def main():
    """메인 실행 함수"""
    print("NEDIS 합성 데이터 독립형 대시보드 생성기")
    print("=" * 60)
    
    # 파일 경로 설정
    current_dir = Path(__file__).parent
    dashboard_dir = current_dir / 'dashboard'
    
    data_file = dashboard_dir / 'serverless_dashboard_data.json'
    template_file = dashboard_dir / 'standalone_interactive_dashboard.html'
    output_file = dashboard_dir / 'nedis_standalone_dashboard.html'
    
    # 파일 존재 여부 확인
    for file_path in [data_file, template_file]:
        if not file_path.exists():
            print(f"오류: 필수 파일이 없습니다: {file_path}")
            sys.exit(1)
    
    try:
        # 1. 데이터 로드
        dashboard_data = load_dashboard_data(data_file)
        
        # 2. 데이터 최적화
        optimized_data = optimize_data_for_embedding(dashboard_data)
        
        # 3. 독립형 대시보드 생성
        output_path = create_embedded_dashboard(template_file, optimized_data, output_file)
        
        # 4. HTML 파일 검증
        validate_html_file(output_path)
        
        # 5. 사용 안내서 생성
        create_demo_instructions(output_path)
        
        print("\n" + "=" * 60)
        print("독립형 대시보드 생성 완료!")
        print(f"파일: {output_path}")
        print(f"크기: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
        print("\n사용 방법:")
        print("1. 브라우저에서 HTML 파일을 직접 열기")
        print("2. 웹 서버에 업로드하여 배포")
        print("3. 오프라인 환경에서도 작동 가능")
        
    except KeyboardInterrupt:
        print("\n프로세스가 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n처리 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()