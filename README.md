# NEDIS Synthetic Data Generation System

National Emergency Department Information System (NEDIS) 합성 데이터 생성 시스템.
원본 응급실 데이터에서 패턴을 학습하여 통계적으로 충실한 합성 데이터를 생성합니다.

---

## Core Principles

- **No Hardcoding** — 모든 분포와 확률은 원본 데이터에서 SQL 기반으로 동적 학습
- **Hierarchical Fallback** — 데이터 희소 시 region → major_region → national → overall 자동 대체
- **Vectorized Generation** — NumPy 벡터화로 322K 레코드 약 7초 생성 (순차 대비 ~50x)
- **Privacy Protection** — K-anonymity, differential privacy, generalization 구현 완료

---

## Architecture

```
Phase 1: Pattern Analysis          Phase 2: Vectorized Generation       Phase 3: Validation
─────────────────────────          ──────────────────────────────       ──────────────────

PatternAnalyzer                    VectorizedPatientGenerator           CorrelationBalanceValidator
  10 pattern types                   Stage 1: Demographics                Cramer's V
  SQL window functions               Stage 2: Hospital assignment          Pearson correlation
  4-level KTAS hierarchy             Stage 3: Independent clinical         Correlation ratio
  Pickle/JSON caching                Stage 4: Conditional clinical
                                                                        SideBySideComparison
                                   TemporalPatternAssigner                Numeric/categorical/temporal
                                     Daily volumes (seasonal+Poisson)     HTML report + JSON metrics
                                     Date: np.searchsorted on CDF
                                     Time: 8-candidate weighted blend   PrivacyValidator
                                                                          K-anonymity, L-diversity
                                   CapacityConstraintPostProcessor        T-closeness, risk scoring
                                     Overflow detection
                                     Same-region redistribution
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- DuckDB database with NEDIS original data

### Setup

```bash
pip install -r requirements.txt
python3 scripts/setup_database.py --db-path nedis_sample.duckdb
```

### Generate Synthetic Data

**Python Pipeline:**

```bash
python3 scripts/run_vectorized_pipeline.py \
  --db-path nedis_sample.duckdb \
  --total-records 322573 \
  --time-resolution hourly
```

**Privacy-Enhanced Generation:**

```bash
python3 scripts/test_enhanced_generation.py \
  --db-path nedis_sample.duckdb \
  --k-anonymity 5 \
  --epsilon 1.0
```

**Iterative Quality Loop:**

```bash
python3 scripts/iterative_synthetic_quality_loop.py \
  --db-path nedis_sample.duckdb \
  --target-overall-score 0.80 \
  --max-iterations 10
```

**Browser-Based Generator:**

```bash
python3 scripts/build_html_generator.py
# outputs/html_generator/nedis_generator.html (standalone, ~256KB)
```

---

## Project Structure

```
src/
  analysis/
    pattern_analyzer.py          # 10-pattern SQL-based dynamic analysis
  vectorized/
    patient_generator.py         # 4-stage vectorized patient generation
    temporal_assigner.py         # Conditional time distribution blending
    capacity_processor.py        # Hospital capacity constraint post-processing
  privacy/
    identifier_manager.py        # Synthetic ID generation
    k_anonymity.py               # K-anonymity validation & enforcement
    generalization.py            # Age/region/temporal generalization
    differential_privacy.py      # Laplace/Gaussian noise, budget accounting
    privacy_validator.py         # Comprehensive privacy risk assessment
  generation/
    enhanced_synthetic_generator.py  # 7-phase privacy-enhanced pipeline
  validation/
    correlation_balance_validator.py # Cramer's V, Pearson, correlation ratio
  comparison/
    visualization.py             # Original vs synthetic side-by-side report

scripts/
  run_vectorized_pipeline.py     # Main generation pipeline
  iterative_synthetic_quality_loop.py  # Generate-validate-adjust loop
  build_html_generator.py        # Build standalone HTML generator
  privacy_risk_assessment.py     # Privacy risk analysis

config/
  generation_params.yaml         # Generation parameters & holiday calendar

templates/
  nedis_generator_template.html  # Browser generator template

reference/
  NEDIS_4.0_CORE_SCHEMA.txt     # NEDIS 4.0 DB schema
  NEDIS_VARIABLE_MAPPING.md     # Sample ↔ NEDIS 4.0 variable mapping
```

---

## Pattern Analysis (10 Types)

| Pattern | Learned Distribution | Fallback |
|---------|---------------------|----------|
| Hospital Allocation | P(hospital \| region) | major_region → national |
| KTAS Distribution | P(KTAS \| region, hospital_type) | 4-level hierarchy |
| Regional | Visit counts, urgency rates per region | — |
| Demographic | P(age_group, sex) joint distribution | — |
| Temporal | Monthly, weekday, hourly distributions | — |
| Temporal Conditional | 8 cross-tabulated hour distributions | global fallback |
| Visit Method | P(vst_meth \| age_group) | — |
| Chief Complaint | P(msypt \| age_group, sex) | — |
| Department | P(main_trt_p \| age_group, sex) | — |
| Treatment Result | P(emtrt_rust \| KTAS, age_group) | — |

---

## Privacy Pipeline

`EnhancedSyntheticGenerator` 7-phase pipeline:

1. **Base Generation** — VectorizedPatientGenerator + TemporalPatternAssigner
2. **Identifier Management** — Direct identifier removal, synthetic ID generation
3. **Generalization** — Age grouping, geographic hierarchy, temporal rounding
4. **K-Anonymity Enforcement** — Suppression or progressive generalization
5. **Differential Privacy** — Laplace/Gaussian noise on vital signs and age
6. **Privacy Validation** — K-anonymity, L-diversity, T-closeness, risk scoring
7. **Post-Processing** — Type normalization, sorting, index reset

---

## Documentation

- [Documentation Index](docs/NEDIS_DOCUMENTATION_INDEX.md)
- [Core Algorithm & Workflows](docs/core_algorithm_and_workflows.md)
- [Pattern Analysis System](docs/sections/03_pattern_analysis_system.md)
- [Synthetic Data Generation](docs/sections/04_synthetic_data_generation.md)
- [Privacy Enhancement Framework](docs/sections/06_privacy_enhancement_framework.md)
- [System Analysis](docs/comprehensive_system_analysis.md)

---

## Output Variables (NEDIS 4.0)

HTML generator outputs use NEDIS 4.0 column IDs:

```
ptmiemcd (emorg_cd)     ptmiindt (vst_dt)      ptmiintm (vst_tm)
ptmibrtd (pat_age_gr)   ptmisexx (pat_sex)     ptmizipc (pat_do_cd)
ptmiinmn (vst_meth)     ptmikts1 (ktas_fstu)   ptmimnsy (msypt)
ptmidept (main_trt_p)   ptmiemrt (emtrt_rust)   ptmiotdt (otrm_dt)
ptmiottm (otrm_tm)      ptmihibp (vst_sbp)     ptmilobp (vst_dbp)
ptmipuls (vst_per_pu)   ptmibrth (vst_per_br)   ptmibdht (vst_bdht)
ptmivoxs (vst_oxy)      ptmihsdt (inpat_dt)    ptmihstm (inpat_tm)
ptmidcrt (inpat_rust)   ptmiakdt (ocur_dt)     ptmiaktm (ocur_tm)
ptmikpr1 (ktas01)       ptmiidno (pat_reg_no)
```

Full mapping: [reference/NEDIS_VARIABLE_MAPPING.md](reference/NEDIS_VARIABLE_MAPPING.md)

---

## License

Private repository. All rights reserved.
