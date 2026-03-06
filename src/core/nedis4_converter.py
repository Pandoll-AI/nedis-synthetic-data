"""
NEDIS 4.0 변수명 변환 모듈 (Single Source of Truth)

2017 표본자료 변수명 <-> NEDIS 4.0 DB 칼럼 ID 간 매핑,
값 변환 함수, DuckDB VIEW DDL 생성을 담당합니다.
"""

import numpy as np
from typing import Dict, Optional, Tuple


# ──────────────────────────────────────────────
# 변수명 매핑 (표본자료 → NEDIS 4.0)
# ──────────────────────────────────────────────

SAMPLE_TO_NEDIS4: Dict[str, str] = {
    'emorg_cd':   'ptmiemcd',
    'pat_reg_no': 'ptmiidno',
    'vst_dt':     'ptmiindt',
    'vst_tm':     'ptmiintm',
    'pat_age_gr': 'ptmibrtd',
    'pat_sex':    'ptmisexx',
    'pat_do_cd':  'ptmizipc',
    'vst_meth':   'ptmiinmn',
    'ktas_fstu':  'ptmikts1',
    'ktas01':     'ptmikpr1',
    'msypt':      'ptmimnsy',
    'main_trt_p': 'ptmidept',
    'emtrt_rust': 'ptmiemrt',
    'otrm_dt':    'ptmiotdt',
    'otrm_tm':    'ptmiottm',
    'ocur_dt':    'ptmiakdt',
    'ocur_tm':    'ptmiaktm',
    'vst_sbp':    'ptmihibp',
    'vst_dbp':    'ptmilobp',
    'vst_per_pu': 'ptmipuls',
    'vst_per_br': 'ptmibrth',
    'vst_bdht':   'ptmibdht',
    'vst_oxy':    'ptmivoxs',
    'inpat_dt':   'ptmihsdt',
    'inpat_tm':   'ptmihstm',
    'inpat_rust': 'ptmidcrt',
}

NEDIS4_TO_SAMPLE: Dict[str, str] = {v: k for k, v in SAMPLE_TO_NEDIS4.items()}


# ──────────────────────────────────────────────
# 값 변환 함수
# ──────────────────────────────────────────────

def convert_sex(value: str) -> str:
    """표본자료 성별 M/F → NEDIS 4.0 코드 1/2"""
    mapping = {'M': '1', 'F': '2'}
    return mapping.get(str(value).upper(), str(value))


def convert_sex_reverse(value: str) -> str:
    """NEDIS 4.0 성별 1/2 → 표본자료 M/F"""
    mapping = {'1': 'M', '2': 'F'}
    return mapping.get(str(value), str(value))


# 시도코드 → 우편번호 범위 매핑
_PROVINCE_ZIP_RANGES: Dict[str, Tuple[int, int]] = {
    '11': (1000, 9999),    # 서울
    '21': (46000, 49999),  # 부산
    '22': (41000, 44999),  # 대구
    '23': (21000, 23999),  # 인천
    '24': (61000, 62999),  # 광주
    '25': (34000, 35999),  # 대전
    '26': (44000, 45999),  # 울산
    '29': (58000, 59999),  # 세종
    '31': (10000, 20999),  # 경기
    '32': (24000, 26999),  # 강원
    '33': (27000, 29999),  # 충북
    '34': (30000, 33999),  # 충남
    '35': (54000, 56999),  # 전북
    '36': (57000, 60999),  # 전남
    '37': (36000, 40999),  # 경북
    '38': (50000, 53999),  # 경남
    '39': (63000, 63999),  # 제주
}


# 연령군 코드 → 연령 범위 매핑 (최소, 최대 나이)
_AGE_GROUP_RANGES: Dict[str, Tuple[int, int]] = {
    '01': (0, 0),      # 영아 (0세)
    '09': (1, 9),      # 유아/소아
    '10': (10, 19),
    '20': (20, 29),
    '30': (30, 39),
    '40': (40, 49),
    '50': (50, 59),
    '60': (60, 69),
    '70': (70, 79),
    '80': (80, 89),
    '90': (90, 99),
    '99': (-1, -1),    # 미상
}


def generate_synthetic_birthdate(
    age_group: str, visit_date: str, rng: Optional[np.random.RandomState] = None
) -> str:
    """연령군 코드 + 내원일자 → 합성 YYYYMMDD 생년월일.

    Args:
        age_group: 연령군 코드 (예: '20', '01', '99')
        visit_date: 내원일자 YYYYMMDD
        rng: numpy RandomState (None이면 기본 사용)

    Returns:
        YYYYMMDD 형식 합성 생년월일, 미상일 경우 '99999999'
    """
    if rng is None:
        rng = np.random.RandomState()

    if age_group == '99' or age_group not in _AGE_GROUP_RANGES:
        return '99999999'

    try:
        visit_year = int(visit_date[:4])
    except (ValueError, TypeError):
        visit_year = 2017

    min_age, max_age = _AGE_GROUP_RANGES[age_group]

    if age_group == '01':
        # 영아: 0세 → 생년은 visit_year-1 ~ visit_year
        birth_year = rng.choice([visit_year - 1, visit_year])
    else:
        # 해당 연령 범위에서 랜덤 나이 선택
        age = rng.randint(min_age, max_age + 1)
        birth_year = visit_year - age

    birth_month = rng.randint(1, 13)
    # 월별 최대 일수
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    # 윤년 처리
    if birth_year % 4 == 0 and (birth_year % 100 != 0 or birth_year % 400 == 0):
        days_in_month[1] = 29
    max_day = days_in_month[birth_month - 1]
    birth_day = rng.randint(1, max_day + 1)

    return f"{birth_year:04d}{birth_month:02d}{birth_day:02d}"


def generate_synthetic_zipcode(
    province_code: str, rng: Optional[np.random.RandomState] = None
) -> str:
    """시도코드 → 12자리 합성 우편번호.

    Args:
        province_code: 2자리 시도코드 (예: '11', '21')
        rng: numpy RandomState

    Returns:
        12자리 문자열. 미상이면 '999999999999'
    """
    if rng is None:
        rng = np.random.RandomState()

    if province_code == '99' or province_code not in _PROVINCE_ZIP_RANGES:
        return '999999999999'

    zip_min, zip_max = _PROVINCE_ZIP_RANGES[province_code]
    zipcode_5 = rng.randint(zip_min, zip_max + 1)
    # 12자리: 5자리 우편번호 + 7자리 패딩 (0)
    return f"{zipcode_5:05d}0000000"


def generate_synthetic_birthdates_vectorized(
    age_groups: np.ndarray, visit_dates: np.ndarray,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """벡터화된 합성 생년월일 생성."""
    if rng is None:
        rng = np.random.RandomState()
    results = np.empty(len(age_groups), dtype='U8')
    for i in range(len(age_groups)):
        results[i] = generate_synthetic_birthdate(
            str(age_groups[i]), str(visit_dates[i]), rng
        )
    return results


def generate_synthetic_zipcodes_vectorized(
    province_codes: np.ndarray,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """벡터화된 합성 우편번호 생성."""
    if rng is None:
        rng = np.random.RandomState()
    results = np.empty(len(province_codes), dtype='U12')
    for i in range(len(province_codes)):
        results[i] = generate_synthetic_zipcode(str(province_codes[i]), rng)
    return results


# ──────────────────────────────────────────────
# DuckDB VIEW DDL 생성
# ──────────────────────────────────────────────

def create_source_view_sql(source_table: str, view_name: str = 'nedis_original.emihptmi') -> str:
    """원본 표본자료 테이블을 NEDIS 4.0 칼럼명으로 매핑하는 VIEW DDL.

    - 칼럼명 alias 변환
    - pat_sex M/F → 1/2 변환
    - pat_age_gr, pat_do_cd 는 코드 유지 (합성 시 변환)

    Args:
        source_table: 원본 테이블 (예: 'nedis_original.nedis2017')
        view_name: 생성할 VIEW 이름

    Returns:
        CREATE OR REPLACE VIEW DDL 문자열
    """
    return f"""
CREATE OR REPLACE VIEW {view_name} AS
SELECT
    emorg_cd   AS ptmiemcd,
    pat_reg_no AS ptmiidno,
    vst_dt     AS ptmiindt,
    vst_tm     AS ptmiintm,
    pat_age_gr AS ptmibrtd,
    CASE WHEN pat_sex = 'M' THEN '1'
         WHEN pat_sex = 'F' THEN '2'
         ELSE pat_sex END AS ptmisexx,
    pat_do_cd  AS ptmizipc,
    vst_meth   AS ptmiinmn,
    ktas_fstu  AS ptmikts1,
    ktas01     AS ptmikpr1,
    msypt      AS ptmimnsy,
    main_trt_p AS ptmidept,
    emtrt_rust AS ptmiemrt,
    otrm_dt    AS ptmiotdt,
    otrm_tm    AS ptmiottm,
    ocur_dt    AS ptmiakdt,
    ocur_tm    AS ptmiaktm,
    vst_sbp    AS ptmihibp,
    vst_dbp    AS ptmilobp,
    vst_per_pu AS ptmipuls,
    vst_per_br AS ptmibrth,
    vst_bdht   AS ptmibdht,
    vst_oxy    AS ptmivoxs,
    inpat_dt   AS ptmihsdt,
    inpat_tm   AS ptmihstm,
    inpat_rust AS ptmidcrt,
    index_key
FROM {source_table}
"""
