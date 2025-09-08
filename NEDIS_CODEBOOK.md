# NEDIS 2017 Dataset Code Book

## Overview
This code book documents all 340 columns from the Korean National Emergency Department Information System (NEDIS) 2017 dataset. The data is organized into 5 tables for optimal storage and querying.

## Table Structure

### Primary Key System
- **index_key**: Composite primary key used across all tables
- **Format**: `emorg_cd + "_" + pat_reg_no + "_" + vst_dt + "_" + vst_tm`
- **Purpose**: Uniquely identifies each emergency department visit

---

## Table 1: nedis2017 (Main Visit Data)
**86 columns + index_key**

### Patient Demographics
| Column | Korean Name | Description | Data Type | Notes |
|--------|-------------|-------------|-----------|-------|
| emorg_cd | 응급의료기관코드 | Emergency medical institution code | VARCHAR | Hospital identifier |
| pat_reg_no | 환자등록번호 | Patient registration number | VARCHAR | Patient identifier |
| pat_brdt | 환자생년월일 | Patient birth date | VARCHAR | Format: YYYYMMDD |
| pat_age | 환자나이 | Patient age | VARCHAR | Age in years, may contain '-' |
| pat_age_gr | 환자연령군 | Patient age group | VARCHAR | Age category codes |
| pat_sex | 환자성별 | Patient gender | VARCHAR | M/F |
| pat_sex_cd | 환자성별코드 | Patient gender code | VARCHAR | Numeric gender code |
| pat_sarea | 환자거주지역 | Patient residential area | VARCHAR | Geographic area code |
| pat_do_cd | 환자시도코드 | Patient province code | VARCHAR | Administrative region |
| pat_gu_cd | 환자시군구코드 | Patient district code | VARCHAR | Sub-administrative region |
| pat_nm | 환자명 | Patient name | VARCHAR | Patient name (anonymized) |

### Visit Information
| Column | Korean Name | Description | Data Type | Notes |
|--------|-------------|-------------|-----------|-------|
| vst_dt | 내원일자 | Visit date | VARCHAR | Format: YYYYMMDD |
| vst_tm | 내원시각 | Visit time | VARCHAR | Format: HHMM |
| flag_stue | 플래그상태 | Flag status | VARCHAR | Record status indicator |
| insp_tp | 검사유형 | Inspection type | VARCHAR | Type of examination |
| ocur_dt | 발생일자 | Occurrence date | VARCHAR | Incident date |
| ocur_tm | 발생시각 | Occurrence time | VARCHAR | Incident time |
| vst_sck | 내원질병 | Visit sickness | VARCHAR | Primary complaint category |
| vst_ity | 내원손상 | Visit injury | VARCHAR | Injury type |
| vst_dmg | 내원손상정도 | Visit damage level | VARCHAR | Injury severity |
| vst_rute | 내원경로 | Visit route | VARCHAR | How patient arrived |
| vst_meth | 내원수단 | Visit method | VARCHAR | Transportation method |
| vst_react | 내원반응 | Visit reaction | VARCHAR | Patient response level |

### Symptoms and Complaints
| Column | Korean Name | Description | Data Type | Notes |
|--------|-------------|-------------|-----------|-------|
| msypt | 주증상 | Main symptom | VARCHAR | Primary symptom code |
| msypt_sn | 주증상순번 | Main symptom sequence | VARCHAR | Symptom priority |
| msypt2 | 주증상2 | Main symptom 2 | VARCHAR | Secondary symptom |
| msypt_sn2 | 주증상순번2 | Main symptom sequence 2 | VARCHAR | Second symptom priority |
| msypt3 | 주증상3 | Main symptom 3 | VARCHAR | Tertiary symptom |
| msypt_sn3 | 주증상순번3 | Main symptom sequence 3 | VARCHAR | Third symptom priority |
| emsypt_yn | 응급증상여부 | Emergency symptom flag | VARCHAR | Y/N emergency indicator |

### Vital Signs
| Column | Korean Name | Description | Data Type | Notes |
|--------|-------------|-------------|-----------|-------|
| vst_sbp | 수축기혈압 | Systolic blood pressure | DOUBLE | mmHg, -1 if not measured |
| vst_dbp | 이완기혈압 | Diastolic blood pressure | DOUBLE | mmHg, -1 if not measured |
| vst_per_pu | 맥박수 | Pulse rate | DOUBLE | beats/min, -1 if not measured |
| vst_per_br | 호흡수 | Breathing rate | DOUBLE | breaths/min, -1 if not measured |
| vst_bdht | 체온 | Body temperature | DOUBLE | Celsius, -1 if not measured |
| vst_oxy | 산소포화도 | Oxygen saturation | DOUBLE | %, -1 if not measured |

### KTAS (Korean Triage and Acuity Scale)
| Column | Korean Name | Description | Data Type | Notes |
|--------|-------------|-------------|-----------|-------|
| ktas_no | KTAS번호 | KTAS number | VARCHAR | Triage level identifier |
| ktas01 | KTAS점수 | KTAS score | DOUBLE | 1-5 acuity score (1=most urgent) |
| ktas_fdiv | KTAS초기분류 | KTAS first division | VARCHAR | Initial triage category |
| ktas_fstu | KTAS초기상태 | KTAS first status | VARCHAR | Initial patient status |
| ktas_fdi_a | KTAS초기분류A | KTAS first division A | VARCHAR | Triage subcategory A |
| ktas_fdi_b | KTAS초기분류B | KTAS first division B | VARCHAR | Triage subcategory B |
| ktas_div_j | KTAS분류판정 | KTAS division judgment | VARCHAR | Final triage decision |
| ktas_lsc | KTAS최종점수 | KTAS last score | VARCHAR | Final triage score |
| ktas_cdiv | KTAS변경분류 | KTAS changed division | VARCHAR | Modified triage category |
| ktas_cstu | KTAS변경상태 | KTAS changed status | VARCHAR | Modified patient status |

### Treatment Information
| Column | Korean Name | Description | Data Type | Notes |
|--------|-------------|-------------|-----------|-------|
| otrm_trt_a | 응급실치료행위 | Emergency room treatment action | VARCHAR | ER treatment code |
| mtrt_er_ce | 주요치료응급실 | Main treatment emergency room | VARCHAR | Primary ER treatment |
| spc_trt_yn | 특수치료여부 | Special treatment flag | VARCHAR | Y/N special care indicator |
| emtrt_rust | 응급치료결과 | Emergency treatment result | VARCHAR | Treatment outcome |
| inpat_rtue | 입원경로 | Inpatient route | VARCHAR | Admission pathway |
| main_trt_p | 주요치료과 | Main treatment department | VARCHAR | Primary treating department |
| otrm_dt | 응급실퇴실일자 | ER discharge date | VARCHAR | Format: YYYYMMDD |
| otrm_tm | 응급실퇴실시각 | ER discharge time | VARCHAR | Format: HHMM |
| inpat_dt | 입원일자 | Admission date | VARCHAR | Format: YYYYMMDD |
| inpat_tm | 입원시각 | Admission time | VARCHAR | Format: HHMM |
| inpat_rust | 입원결과 | Admission result | VARCHAR | Admission outcome |
| otpat_dt | 외래일자 | Outpatient date | VARCHAR | Format: YYYYMMDD |
| otpat_tm | 외래시각 | Outpatient time | VARCHAR | Format: HHMM |
| otpat_sn | 외래순번 | Outpatient sequence | VARCHAR | Outpatient visit order |
| otpat_grd | 외래등급 | Outpatient grade | VARCHAR | Outpatient priority level |

### Care Details
| Column | Korean Name | Description | Data Type | Notes |
|--------|-------------|-------------|-----------|-------|
| taip | 치료후상태 | Post-treatment condition | VARCHAR | Patient status after treatment |
| care_bt | 치료혈액검사 | Care blood test | VARCHAR | Blood work indicator |
| care_by | 치료혈액형 | Care blood type | VARCHAR | Blood type testing |
| care_fair | 치료공정 | Care fairness | VARCHAR | Care equity measure |
| care_sair | 치료특별공기 | Care special air | VARCHAR | Special respiratory care |
| care_ht | 치료심박 | Care heart rate | VARCHAR | Cardiac monitoring |
| care_gd | 치료혈당 | Care glucose | VARCHAR | Blood sugar monitoring |
| care_vt | 치료활력징후 | Care vital signs | VARCHAR | Vital signs monitoring |
| care_nowea | 치료비착용 | Care non-wearing | VARCHAR | Non-attached monitoring |
| care_noip | 치료비침습 | Care non-invasive | VARCHAR | Non-invasive procedures |
| care_non | 치료없음 | Care none | VARCHAR | No special care |

### Organization and Transfer
| Column | Korean Name | Description | Data Type | Notes |
|--------|-------------|-------------|-----------|-------|
| mcorg_cd | 의료기관코드 | Medical institution code | VARCHAR | Healthcare facility code |
| trd_org_tp | 전원기관유형 | Transfer organization type | VARCHAR | Referring facility type |
| trd_org_cd | 전원기관코드 | Transfer organization code | VARCHAR | Referring facility code |
| trs_org_tp | 이송기관유형 | Transport organization type | VARCHAR | Transport service type |
| trs_org_cd | 이송기관코드 | Transport organization code | VARCHAR | Transport service code |

### Administrative Fields
| Column | Korean Name | Description | Data Type | Notes |
|--------|-------------|-------------|-----------|-------|
| ca | 구분자 | Category identifier | VARCHAR | Record category |
| key | 키값 | Key value | VARCHAR | Internal key |
| base | 기준 | Base reference | VARCHAR | Reference baseline |
| hospname | 병원명 | Hospital name | VARCHAR | Institution name |
| gubun | 구분 | Division | VARCHAR | Classification code |
| adr | 주소 | Address | VARCHAR | Location information |
| etc | 기타 | Others | VARCHAR | Additional information |
| vr_dt | 검증일자 | Verification date | VARCHAR | Data validation date |
| emorg | 응급기관 | Emergency organization | VARCHAR | Emergency service code |

---

## Table 2: iciss (ICISS Injury Severity Codes)
**Normalized from 32 columns to rows**

### Structure
| Column | Description | Data Type | Notes |
|--------|-------------|-----------|-------|
| index_key | Visit identifier | VARCHAR | Foreign key to nedis2017 |
| position | Code position | INTEGER | 1-32 indicating d01-d28 and ICISS codes |
| code_type | Type of code | VARCHAR | d01-d28, iciss0, iciss1, icis_a, icis_b, srr |
| code_value | Actual code | VARCHAR | The diagnosis or severity code |

### Original Columns Being Normalized
- **d01-d28**: Diagnosis codes 1-28
- **d05_iciss0**: ICISS score 0
- **d05_iciss1**: ICISS score 1  
- **d05_icis_a**: ICISS category A
- **d05_icis_b**: ICISS category B
- **d05_srr**: Survival risk ratio

---

## Table 3: diag_er (Emergency Room Diagnoses)
**Normalized from 122 columns to rows**

### Structure
| Column | Description | Data Type | Notes |
|--------|-------------|-----------|-------|
| index_key | Visit identifier | VARCHAR | Foreign key to nedis2017 |
| position | Diagnosis position | INTEGER | 1-61 indicating diagnosis order |
| diagnosis_code | Diagnosis code | VARCHAR | From dgotdiag01-61 |
| diagnosis_category | Diagnosis category | VARCHAR | From dgotdggb01-61 |

### Original Column Pattern
- **dgotdiag01-dgotdiag61**: Emergency room diagnosis codes
- **dgotdggb01-dgotdggb61**: Emergency room diagnosis categories

---

## Table 4: diag_adm (Admission Diagnoses)
**Normalized from 100 columns to rows**

### Structure
| Column | Description | Data Type | Notes |
|--------|-------------|-----------|-------|
| index_key | Visit identifier | VARCHAR | Foreign key to nedis2017 |
| position | Diagnosis position | INTEGER | 1-50 indicating diagnosis order |
| diagnosis_code | Diagnosis code | VARCHAR | From dgdcdiag01-50 |
| diagnosis_category | Diagnosis category | VARCHAR | From dgdcdggb01-50 |

### Original Column Pattern
- **dgdcdiag01-dgdcdiag50**: Admission diagnosis codes
- **dgdcdggb01-dgdcdggb50**: Admission diagnosis categories

---

## Data Type Analysis Results
*Based on analysis of nedis_1.h5 (100,000 records)*

### Optimal DuckDB Data Types

#### INTEGER Columns (6 columns)
- **ktas01**: KTAS triage score (1-5 scale, some 6-8 values exist)
- **vst_sbp**: Systolic blood pressure (mmHg, -1 for not measured)
- **vst_dbp**: Diastolic blood pressure (mmHg, -1 for not measured)
- **vst_per_pu**: Pulse rate (beats/min, -1 for not measured)
- **vst_per_br**: Breathing rate (breaths/min, -1 for not measured)
- **vst_oxy**: Oxygen saturation (%, -1 for not measured)

#### VARCHAR Columns (80 columns)
All other main table columns should be VARCHAR due to:
- Mixed data formats (dates as YYYYMMDD strings)
- Special values like '-' in numeric fields
- Korean text and codes requiring UTF-8 support

### Common Value Patterns

#### Hospital Distribution
- **이화여자대학교의과대학부속목동병원**: 48,699 visits (48.7%)
- **순천향대학교부속서울병원**: 36,994 visits (37.0%)  
- **중앙대학교병원**: 14,307 visits (14.3%)

#### Patient Demographics
- **Gender**: Female 51,079 (51.1%), Male 48,921 (48.9%)
- **Age**: Peak at 1 year (5,392 visits), decreasing with age
- **Age format**: "001.00" format (3 digits + ".00")

#### KTAS Triage Distribution
- **Level 4**: 42,448 visits (42.4%) - Less urgent
- **Level 3**: 40,018 visits (40.0%) - Urgent  
- **Level 5**: 10,415 visits (10.4%) - Non-urgent
- **Level 2**: 4,611 visits (4.6%) - Emergent
- **Level 1**: 2,508 visits (2.5%) - Immediate

#### Vital Signs Measurement Rates
- **Blood Pressure**: ~80% measured (20% show -1)
- **Temperature**: Nearly all measured
- **Pulse/Breathing**: ~80% measured
- **Oxygen Saturation**: Lower measurement rate

### Diagnosis Data Statistics
- **ER Diagnoses**: Average 2.0 diagnoses per visit (dgotdiag01-61)
- **Admission Diagnoses**: Average 0.5 diagnoses per visit (dgdcdiag01-50)
- **ICISS Codes**: Sparsely populated, mostly empty

## Data Quality Notes

### Missing Value Indicators
- **Numeric fields**: -1 indicates "not measured" or "not applicable"
- **Text fields**: Empty strings, 'nan', or null values indicate missing data
- **Date/Time fields**: Empty if event did not occur (e.g., inpat_dt for outpatients)
- **Diagnosis codes**: Empty cells indicate no additional diagnoses

### Data Validation Findings
- **Complete coverage**: All 100,000 records have core fields populated
- **Consistent formatting**: Dates as YYYYMMDD, times as HHMM
- **Unique identifiers**: 85,355 unique patients (some repeat visits)
- **Hospital codes**: 3 major hospitals represented consistently

### Key Relationships
- **One-to-Many**: One visit (nedis2017) can have multiple diagnoses (diag_er, diag_adm) and ICISS codes (iciss)
- **Foreign Key**: index_key links all tables
- **Referential Integrity**: All diagnosis and ICISS records must have corresponding nedis2017 record

---

## Data Sources and Methodology
- **Source**: Korean National Emergency Department Information System (NEDIS) 2017
- **Coverage**: Emergency department visits in South Korea
- **Time Period**: Calendar year 2017
- **File Format**: HDF5 files (nedis_0.h5 to nedis_91.h5)
- **Total Records**: Approximately 9.2 million emergency department visits

---

*This code book will be updated with specific data type analysis and common value findings after examining the actual data files.*