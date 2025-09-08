#!/usr/bin/env python3
"""
대시보드 테스트를 위한 샘플 역학 데이터 생성 스크립트
"""

import json
from datetime import datetime

def create_sample_epidemiologic_data():
    """테스트용 샘플 역학 데이터 생성"""
    
    sample_data = {
        "metadata": {
            "analysis_date": datetime.now().isoformat(),
            "sample_db": "nedis_sample.duckdb",
            "synthetic_db": "nedis_synthetic.duckdb",
            "note": "Sample data for dashboard testing"
        },
        "demographics": {
            "age_distribution": {
                "sample": [
                    {"age_group": "0-9세", "count": 4404, "percentage": 1.37},
                    {"age_group": "10-19세", "count": 25712, "percentage": 7.97},
                    {"age_group": "20-29세", "count": 20483, "percentage": 6.35},
                    {"age_group": "30-39세", "count": 48029, "percentage": 14.89},
                    {"age_group": "40-49세", "count": 39379, "percentage": 12.21},
                    {"age_group": "50-59세", "count": 36153, "percentage": 11.21},
                    {"age_group": "60-69세", "count": 44350, "percentage": 13.75},
                    {"age_group": "70-79세", "count": 52142, "percentage": 16.16},
                    {"age_group": "80세 이상", "count": 51948, "percentage": 16.10}
                ]
            },
            "sex_distribution": {
                "sample": [
                    {"sex": "남성", "count": 162300, "percentage": 50.3},
                    {"sex": "여성", "count": 160300, "percentage": 49.7}
                ]
            },
            "region_distribution": {
                "sample": [
                    {"region_code": "서울", "count": 85000, "percentage": 26.3},
                    {"region_code": "경기", "count": 72000, "percentage": 22.3},
                    {"region_code": "부산", "count": 34000, "percentage": 10.5},
                    {"region_code": "대구", "count": 28000, "percentage": 8.7},
                    {"region_code": "인천", "count": 25000, "percentage": 7.7},
                    {"region_code": "광주", "count": 18000, "percentage": 5.6},
                    {"region_code": "대전", "count": 20000, "percentage": 6.2},
                    {"region_code": "울산", "count": 12000, "percentage": 3.7},
                    {"region_code": "세종", "count": 3500, "percentage": 1.1},
                    {"region_code": "기타", "count": 25100, "percentage": 7.9}
                ]
            }
        },
        "disease_epidemiology": {
            "ktas_distribution": {
                "sample": [
                    {"ktas_no": "1", "count": 8500, "percentage": 2.6},
                    {"ktas_no": "2", "count": 32500, "percentage": 10.1},
                    {"ktas_no": "3", "count": 127800, "percentage": 39.6},
                    {"ktas_no": "4", "count": 118200, "percentage": 36.6},
                    {"ktas_no": "5", "count": 35600, "percentage": 11.0}
                ]
            }
        },
        "temporal_epidemiology": {
            "monthly_pattern": {
                "sample": [
                    {"month": 1, "count": 28500},
                    {"month": 2, "count": 26200},
                    {"month": 3, "count": 27800},
                    {"month": 4, "count": 26900},
                    {"month": 5, "count": 27200},
                    {"month": 6, "count": 26800},
                    {"month": 7, "count": 28100},
                    {"month": 8, "count": 27500},
                    {"month": 9, "count": 26700},
                    {"month": 10, "count": 27400},
                    {"month": 11, "count": 26300},
                    {"month": 12, "count": 28200}
                ]
            },
            "weekday_pattern": {
                "sample": [
                    {"weekday": "월요일", "total_count": 46200},
                    {"weekday": "화요일", "total_count": 45800},
                    {"weekday": "수요일", "total_count": 45500},
                    {"weekday": "목요일", "total_count": 46100},
                    {"weekday": "금요일", "total_count": 47300},
                    {"weekday": "토요일", "total_count": 48900},
                    {"weekday": "일요일", "total_count": 42800}
                ]
            },
            "hourly_pattern": {
                "sample": [
                    {"hour": h, "count": 8000 + (h * 100) + (abs(12 - h) * 50)} 
                    for h in range(24)
                ]
            }
        },
        "clinical_epidemiology": {
            "vital_signs": {
                "sample": {
                    "avg_sbp": 125.5,
                    "avg_dbp": 78.2,
                    "avg_pulse": 82.1,
                    "avg_respiration": 18.5,
                    "avg_oxygen": 97.8,
                    "sbp_records": 280000,
                    "dbp_records": 280000,
                    "pulse_records": 295000,
                    "respiration_records": 275000,
                    "oxygen_records": 290000,
                    "total_records": 322600
                }
            },
            "ktas_vital_patterns": {
                "sample": [
                    {"ktas_no": "1", "avg_sbp": 95.2, "avg_pulse": 110.5, "avg_oxygen": 85.2},
                    {"ktas_no": "2", "avg_sbp": 115.8, "avg_pulse": 98.3, "avg_oxygen": 92.1},
                    {"ktas_no": "3", "avg_sbp": 125.4, "avg_pulse": 84.7, "avg_oxygen": 97.5},
                    {"ktas_no": "4", "avg_sbp": 128.2, "avg_pulse": 78.9, "avg_oxygen": 98.7},
                    {"ktas_no": "5", "avg_sbp": 130.5, "avg_pulse": 75.2, "avg_oxygen": 99.1}
                ]
            }
        },
        "spatial_epidemiology": {
            "hospital_distribution": {
                "sample": [
                    {"hospital_code": f"H{str(i).zfill(6)}", "patient_count": 15000 - (i * 500)} 
                    for i in range(1, 21)
                ]
            }
        }
    }
    
    return sample_data

def main():
    """메인 함수"""
    print("Creating sample epidemiologic data for dashboard testing...")
    
    sample_data = create_sample_epidemiologic_data()
    
    # 샘플 데이터를 JSON 파일로 저장
    output_path = "outputs/epidemiologic_analysis_sample.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"Sample data saved to: {output_path}")
    print(f"File size: {len(json.dumps(sample_data, ensure_ascii=False)) / 1024:.1f} KB")
    print("\nYou can now test the dashboard with this smaller sample file.")

if __name__ == "__main__":
    main()