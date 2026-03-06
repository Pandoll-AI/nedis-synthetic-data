# NEDIS 변수명 매칭 레퍼런스

## 개요
- **표본자료 변수명 (2017)**: `NEDIS_CODEBOOK.md`에 정의된 연구자 배포용 변수명
- **NEDIS 4.0 DB 칼럼 ID**: `reference/NEDIS_4.0_CORE_SCHEMA.txt`에 정의된 실제 DB 스키마 (EMIHPTMI 테이블, `ptmi` 접두사)
- **NEDIS 4.0 입력지침**: `reference/NEDIS_4.0_DETAIL_INSTRUCTION.txt`에 각 변수별 코드값, 입력규칙, 유효성 검증 조건 상세 기술

## HTML 생성기 출력 변수 매칭표 (27개)

| # | 표본자료 변수명 (HTML 출력) | NEDIS 4.0 칼럼 ID | 한글명 | 비고 |
|---|---|---|---|---|
| 1 | `index_key` | — | 색인키 | 복합키 (emorg_cd + pat_reg_no + vst_dt + vst_tm) |
| 2 | `emorg_cd` | `ptmiemcd` | 응급의료기관코드 | 8자리, 중앙응급의료센터 부여 |
| 3 | `pat_reg_no` | `ptmiidno` | 의무기록번호 | 12자리, 비식별화 처리 |
| 4 | `vst_dt` | `ptmiindt` | 내원일자 | YYYYMMDD |
| 5 | `vst_tm` | `ptmiintm` | 내원시간 | HH24MM |
| 6 | `pat_age_gr` | `ptmibrtd` (변환) | 환자연령군 | DB는 생년월일, 표본자료는 연령군 코드로 변환 |
| 7 | `pat_sex` | `ptmisexx` | 성별 | DB: 주민번호 7번째 자리(N,1), 표본: M/F |
| 8 | `pat_do_cd` | `ptmizipc` | 환자시도코드 | DB: 도로명주소(12자리), 표본: 시도코드 추출 |
| 9 | `vst_meth` | `ptmiinmn` | 내원수단 | C, 2자리 코드 |
| 10 | `ktas_fstu` | `ptmikts1` | 최초 중증도 분류 결과 | C, 1자리 (1~5) |
| 11 | `ktas01` | `ptmikpr1` | 최초 중증도 분류 과정 | C, 6자리 |
| 12 | `msypt` | `ptmimnsy` | 주증상 | UMLS 기반 주증상 코드, C, 8자리 |
| 13 | `main_trt_p` | `ptmidept` | 주요치료과 / 주 진료과 | C, 2자리 코드 |
| 14 | `emtrt_rust` | `ptmiemrt` | 응급진료결과 | C, 2자리 (귀가/전원/입원/사망/기타/미상) |
| 15 | `otrm_dt` | `ptmiotdt` | 퇴실일자 | YYYYMMDD |
| 16 | `otrm_tm` | `ptmiottm` | 퇴실시간 | HH24MM |
| 17 | `ocur_dt` | `ptmiakdt` | 발병일자 | YYYYMMDD (미상: 11111111) |
| 18 | `ocur_tm` | `ptmiaktm` | 발병시간 | HH24MM (미상: 1111) |
| 19 | `vst_sbp` | `ptmihibp` | 수축기 혈압 | N, 3자리 (mmHg), 미측정: -1, default: 999 |
| 20 | `vst_dbp` | `ptmilobp` | 이완기 혈압 | N, 3자리 (mmHg), 미측정: -1, default: 999 |
| 21 | `vst_per_pu` | `ptmipuls` | 맥박수 | N, 3자리 (회/분), 미측정: -1, default: 999 |
| 22 | `vst_per_br` | `ptmibrth` | 호흡수 | N, 3자리 (회/분), 미측정: -1, default: 999 |
| 23 | `vst_bdht` | `ptmibdht` | 체온 | N, 3.1자리 (℃), 미측정: -1, default: 99.9 |
| 24 | `vst_oxy` | `ptmivoxs` | 산소포화도 | N, 3자리 (%), 미측정: -1, default: 999 |
| 25 | `inpat_dt` | `ptmihsdt` | 입원일자 | YYYYMMDD |
| 26 | `inpat_tm` | `ptmihstm` | 입원시간 | HH24MM |
| 27 | `inpat_rust` | `ptmidcrt` | 입원 후 결과 | C, 2자리 |

## NEDIS 4.0 EMIHPTMI 테이블 전체 필드 (표본자료에 미포함된 항목 포함)

표본자료에 포함되지 않지만 NEDIS 4.0 DB에 존재하는 주요 필드:

| NEDIS 4.0 칼럼 ID | 한글명 | 미포함 사유 |
|---|---|---|
| `ptmistat` | 자료처리상태 | 내부 운영용 |
| `ptminame` | 성명 | 개인정보 (비식별 항목) |
| `ptmiiukd` | 보험유형 | 2자리 코드 |
| `ptmihscd` | 요양기관번호 | 8자리 |
| `ptmidrlc` | 진료의사 면허번호 | 개인정보 |
| `ptmidgkd` | 질병여부 | 1자리 (1:질병, 2:질병외, 3:진료외방문 등) |
| `ptmiarcf` | 의도성 여부 | 질병외인 경우만 |
| `ptmiarcs` | 손상기전 | 질병외인 경우만 |
| `ptmiinrt` | 내원경로 | 1자리 코드 |
| `ptmiemsy` | 응급증상 해당 여부 | Y/N |
| `ptmiresp` | 내원 시 반응 | A/V/P/U |
| `ptmiarea` | 최종진료구역 | 1자리 |
| `ptmisdcd` | 전문의 진료 여부 | 1자리 |
| `ptmiktid` | 중증도 분류 선별번호 | 12자리 |
| `ptmikpr2` | 변경된 중증도 분류과정 | 6자리 |
| `ptmikts2` | 변경된 중증도 분류결과 | 1자리 |
| `ptmihsrt` | 입원경로 | 2자리 |
| `ptmidcdt` | 퇴원일자 | YYYYMMDD |
| `ptmidctm` | 퇴원시간 | HH24MM |

## 참고: NEDIS 4.0 테이블 구조

| 테이블명 | 내역 | 비고 |
|---|---|---|
| EMIHPTMI | 응급환자 진료내역 | 핵심 테이블 (위 매칭표 대상) |
| EMIHTRPT | 응급환자 검사/처치/수술 내역 | EDI 수가코드 사용 |
| EMIHDGOT | 응급환자 퇴실 시 진단 내역 | KCD 코드 (주진단/부진단/의증) |
| EMIHSDMD | 응급환자 전문의 진료내역 | |
| EMIHOPPT | 입원 시 검사/처치/수술 내역 | 응급실 경유 입원 |
| EMIHDGDC | 응급환자 퇴원 시 진단 내역 | 응급실 경유 입원 후 퇴원 |

## 데이터 규칙 요약

- Primary Key: `ptmiemcd` + `ptmiidno` + `ptmiindt` + `ptmiintm`
- NN (Not Null) 필드에 전송할 내용 없을 시: 문자형 `'-'`, 숫자형 바이트수만큼 `'9'` (1바이트→9, 2바이트→99, ...)
- 활력징후 측정불가 또는 비대상: `-1`로 처리
- 발병일시 모두 미상: 발병일자 `'11111111'`, 발병시간 `'1111'`
