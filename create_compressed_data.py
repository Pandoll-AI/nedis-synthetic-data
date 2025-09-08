#!/usr/bin/env python3
"""
대용량 JSON 파일을 압축하고 최적화하는 스크립트
"""

import json
import gzip
import os
from pathlib import Path

def compress_json_data(input_path: str, output_dir: str = "dashboard"):
    """JSON 파일을 gzip으로 압축"""
    
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Input file not found: {input_path}")
        return False
    
    # 출력 디렉터리 생성
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 압축 파일 경로
    compressed_file = output_path / f"{input_file.stem}.json.gz"
    
    try:
        print(f"Compressing {input_path}...")
        
        # 원본 파일 크기 확인
        original_size = input_file.stat().st_size
        print(f"Original file size: {original_size / (1024*1024):.1f} MB")
        
        # JSON을 gzip으로 압축
        with open(input_file, 'rb') as f_in:
            with gzip.open(compressed_file, 'wb') as f_out:
                f_out.write(f_in.read())
        
        # 압축된 파일 크기 확인
        compressed_size = compressed_file.stat().st_size
        compression_ratio = (1 - compressed_size / original_size) * 100
        
        print(f"Compressed file size: {compressed_size / (1024*1024):.1f} MB")
        print(f"Compression ratio: {compression_ratio:.1f}%")
        print(f"Compressed file saved: {compressed_file}")
        
        return True
        
    except Exception as e:
        print(f"Error compressing file: {e}")
        return False

def create_optimized_data(input_path: str, output_dir: str = "dashboard"):
    """대용량 JSON을 최적화된 형태로 변환"""
    
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Input file not found: {input_path}")
        return False
    
    try:
        print(f"Loading and optimizing {input_path}...")
        
        # JSON 로드
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 최적화된 버전 생성 (상위 N개 항목만 포함)
        optimized_data = optimize_data_structure(data)
        
        # 최적화된 데이터를 저장
        output_path = Path(output_dir) / "epidemiologic_analysis_optimized.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(optimized_data, f, ensure_ascii=False, separators=(',', ':'))
        
        # 파일 크기 비교
        original_size = input_file.stat().st_size
        optimized_size = output_path.stat().st_size
        reduction = (1 - optimized_size / original_size) * 100
        
        print(f"Original size: {original_size / (1024*1024):.1f} MB")
        print(f"Optimized size: {optimized_size / (1024*1024):.1f} MB")
        print(f"Size reduction: {reduction:.1f}%")
        print(f"Optimized file saved: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error optimizing file: {e}")
        return False

def optimize_data_structure(data):
    """데이터 구조를 최적화 (대시보드 표시에 필요한 상위 항목만 유지)"""
    
    optimized = {
        "metadata": data.get("metadata", {}),
        "demographics": {},
        "disease_epidemiology": {},
        "temporal_epidemiology": {},
        "clinical_epidemiology": {},
        "spatial_epidemiology": {}
    }
    
    # Demographics 최적화
    demographics = data.get("demographics", {})
    if "age_distribution" in demographics:
        optimized["demographics"]["age_distribution"] = demographics["age_distribution"]
    if "sex_distribution" in demographics:
        optimized["demographics"]["sex_distribution"] = demographics["sex_distribution"]
    if "region_distribution" in demographics and demographics["region_distribution"].get("sample"):
        # 상위 20개 지역만 유지
        region_data = demographics["region_distribution"]["sample"][:20]
        optimized["demographics"]["region_distribution"] = {"sample": region_data}
    
    # Disease epidemiology 최적화
    disease = data.get("disease_epidemiology", {})
    if "ktas_distribution" in disease:
        optimized["disease_epidemiology"]["ktas_distribution"] = disease["ktas_distribution"]
    
    # Temporal epidemiology 최적화
    temporal = data.get("temporal_epidemiology", {})
    for key in ["monthly_pattern", "weekday_pattern", "hourly_pattern"]:
        if key in temporal:
            optimized["temporal_epidemiology"][key] = temporal[key]
    
    # Clinical epidemiology 최적화
    clinical = data.get("clinical_epidemiology", {})
    if "vital_signs" in clinical:
        optimized["clinical_epidemiology"]["vital_signs"] = clinical["vital_signs"]
    if "ktas_vital_patterns" in clinical:
        optimized["clinical_epidemiology"]["ktas_vital_patterns"] = clinical["ktas_vital_patterns"]
    
    # Spatial epidemiology 최적화 
    spatial = data.get("spatial_epidemiology", {})
    if "hospital_distribution" in spatial and spatial["hospital_distribution"].get("sample"):
        # 상위 20개 병원만 유지
        hospital_data = spatial["hospital_distribution"]["sample"][:20]
        optimized["spatial_epidemiology"]["hospital_distribution"] = {"sample": hospital_data}
    
    return optimized

def main():
    """메인 함수"""
    input_file = "outputs/epidemiologic_analysis.json"
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        print("Please run the epidemiologic analysis first.")
        return
    
    print("Creating optimized and compressed versions of epidemiologic data...")
    
    # 1. 최적화된 버전 생성
    success_opt = create_optimized_data(input_file)
    
    # 2. 압축 버전 생성
    success_comp = compress_json_data(input_file)
    
    if success_opt or success_comp:
        print("\nOptimization complete!")
        print("\nYou can now use:")
        print("- epidemiologic_analysis_optimized.json (reduced size)")
        print("- epidemiologic_analysis.json.gz (compressed)")
        print("\nUpdate the dashboard to use these optimized versions.")
    else:
        print("Optimization failed.")

if __name__ == "__main__":
    main()