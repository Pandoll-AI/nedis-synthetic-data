#!/usr/bin/env python3
"""
NEDIS 역학 분석 대시보드 서버

인터랙티브 HTML 대시보드를 실행하기 위한 로컬 웹 서버입니다.
"""

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path
import argparse
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """대시보드를 위한 커스텀 HTTP 핸들러"""
    
    def __init__(self, *args, **kwargs):
        # 프로젝트 루트 디렉토리로 설정
        super().__init__(*args, directory=str(Path(__file__).parent.parent), **kwargs)
    
    def end_headers(self):
        # CORS 헤더 추가
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_GET(self):
        """GET 요청 처리"""
        # 루트 경로로 접근 시 대시보드로 리다이렉트
        if self.path == '/' or self.path == '':
            self.path = '/dashboard/epidemiologic_dashboard.html'
        
        # JSON 파일에 대한 올바른 Content-Type 설정
        if self.path.endswith('.json'):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            try:
                file_path = Path(self.directory) / self.path.lstrip('/')
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        self.wfile.write(f.read())
                else:
                    self.send_error(404, "File not found")
            except Exception as e:
                logger.error(f"Error serving file {self.path}: {e}")
                self.send_error(500, "Internal server error")
        else:
            super().do_GET()
    
    def log_message(self, format, *args):
        """로그 메시지 커스터마이징"""
        logger.info(f"{self.address_string()} - {format % args}")


def check_required_files():
    """필수 파일들이 존재하는지 확인"""
    project_root = Path(__file__).parent.parent
    required_files = [
        'dashboard/epidemiologic_dashboard.html'
    ]
    
    # 데이터 파일 중 하나라도 있으면 OK
    data_files = [
        'dashboard/epidemiologic_analysis_sample.json',
        'dashboard/epidemiologic_analysis_optimized.json', 
        'outputs/epidemiologic_analysis.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    # 데이터 파일 확인 
    data_file_exists = False
    available_data_files = []
    for file_path in data_files:
        full_path = project_root / file_path
        if full_path.exists():
            data_file_exists = True
            file_size = full_path.stat().st_size
            if file_size < 1024*1024:  # < 1MB
                size_str = f"{file_size/1024:.1f}KB"
            else:
                size_str = f"{file_size/(1024*1024):.1f}MB"
            available_data_files.append(f"{file_path} ({size_str})")
    
    if missing_files:
        logger.error("다음 필수 파일들이 누락되었습니다:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        return False
    
    if not data_file_exists:
        logger.error("데이터 파일을 찾을 수 없습니다. 다음 중 하나를 실행하세요:")
        logger.info("  1. 샘플 데이터: python create_sample_data.py")
        logger.info("  2. 역학 분석: python -c \"from src.analysis.epidemiologic_analyzer import main; main()\"")
        logger.info("  3. 데이터 최적화: python create_compressed_data.py")
        return False
    
    logger.info("사용 가능한 데이터 파일:")
    for file_info in available_data_files:
        logger.info(f"  ✓ {file_info}")
    
    return True


def run_dashboard(port=8000, host='localhost', open_browser=True):
    """대시보드 서버 실행"""
    
    # 필수 파일 확인
    if not check_required_files():
        logger.error("필수 파일들이 누락되어 서버를 시작할 수 없습니다.")
        return False
    
    try:
        # 서버 시작
        with socketserver.TCPServer((host, port), DashboardHandler) as httpd:
            url = f"http://{host}:{port}"
            
            logger.info(f"NEDIS 역학 분석 대시보드가 시작되었습니다.")
            logger.info(f"URL: {url}")
            logger.info(f"종료하려면 Ctrl+C를 누르세요.")
            
            # 브라우저 자동 열기
            if open_browser:
                try:
                    webbrowser.open(url)
                    logger.info("브라우저가 자동으로 열렸습니다.")
                except Exception as e:
                    logger.warning(f"브라우저를 자동으로 열 수 없습니다: {e}")
                    logger.info(f"수동으로 브라우저에서 {url}에 접속하세요.")
            
            # 서버 실행
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        logger.info("서버가 종료되었습니다.")
        return True
    except OSError as e:
        if e.errno == 48:  # Address already in use
            logger.error(f"포트 {port}가 이미 사용 중입니다. 다른 포트를 사용하세요.")
        else:
            logger.error(f"서버 시작 중 오류 발생: {e}")
        return False
    except Exception as e:
        logger.error(f"예상치 못한 오류 발생: {e}")
        return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="NEDIS 역학 분석 대시보드 서버",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
    python scripts/run_dashboard.py                    # 기본 설정으로 실행
    python scripts/run_dashboard.py --port 8000        # 포트 8000으로 실행
    python scripts/run_dashboard.py --no-browser       # 브라우저 자동 실행 안함
    python scripts/run_dashboard.py --host 0.0.0.0     # 외부 접속 허용
        """
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=8000,
        help='웹 서버 포트 (기본값: 8000)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='웹 서버 호스트 (기본값: localhost)'
    )
    
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='브라우저 자동 실행 비활성화'
    )
    
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='필수 파일만 확인하고 종료'
    )
    
    args = parser.parse_args()
    
    # 파일 확인만 하고 종료
    if args.check_only:
        if check_required_files():
            logger.info("모든 필수 파일이 존재합니다.")
            return 0
        else:
            return 1
    
    # 대시보드 실행
    success = run_dashboard(
        port=args.port,
        host=args.host,
        open_browser=not args.no_browser
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())