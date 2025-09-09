# Revision v0.3 — Vectorized Pipeline Improvements

Date: 2025-09-09

This revision aligns the implementation with the vectorized generation design and removes hard-coded parameters by shifting to dynamic, data-driven distributions with config-backed fallbacks.

## Summary of Changes

- Dynamic attribute distributions:
  - Added analyzers for P(vst_meth | age), P(msypt | age, sex), P(main_trt_p | age, sex), P(emtrt_rust | KTAS, age).
  - Replaced all hard-coded sampling in patient generation with these learned patterns.

- Batch KTAS generation:
  - Rewrote KTAS assignment to operate in grouped batches by (region, hospital_type) using hierarchical distributions, removing per-row loops.

- Capacity redistribution:
  - Implemented same-region-first reassignment (no distance fallback). Removed distance-matrix usage and the nearest-within-50km option.

- Quality gate:
  - Added optional quality gate that runs Statistical and Clinical validations; pipeline fails if the combined score falls below a threshold (default 0.95).

- CLI defaults and options:
  - Batch size default set to 100,000 and memory-efficient mode enabled by default.
  - Overflow method choices updated to [random_available, same_region_first, second_choice_probability].

- Hard-coded assumptions removed:
  - Hourly fallback distribution moved to config (`temporal.fallback_hour_weights`).
  - Attribute fallback distributions moved to config (`fallback.distributions`).

## Touched Modules

- `src/analysis/pattern_analyzer.py`
  - New analyzers: visit_method_patterns, chief_complaint_patterns, department_patterns, treatment_result_patterns.
- `src/vectorized/patient_generator.py`
  - Independent attributes now sampled from dynamic distributions.
  - KTAS assignment now grouped/batch-based; treatment outcomes use learned conditional distributions.
- `src/vectorized/temporal_assigner.py`
  - Fallback hourly time distribution pulled from config (no hard-coding).
- `src/vectorized/capacity_processor.py`
  - Same-region-first redistribution; removed distance-based logic.
- `scripts/run_vectorized_pipeline.py`
  - CLI defaults updated; quality gate integrated; overflow method options updated.
- `src/core/config.py`
  - Added fallback distributions and temporal fallback hour weights.

## Migration Notes

- If running without original data tables, configure `fallback.distributions` and `temporal.fallback_hour_weights` in `config/generation_params.yaml` to ensure reasonable sampling.
- If you previously relied on `nearest_available`, use `same_region_first` or `second_choice_probability` instead.

