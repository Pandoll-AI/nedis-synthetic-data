# Core Algorithm and Workflows

This document summarizes the vectorized synthetic data generation system — its core algorithmic components and end-to-end workflows.

## Principles

- Time separation: Generate all patient attributes independent of time; assign dates/times afterwards.
- Dynamic patterns: Learn all distributions directly from original data, cached for reuse.
- Hierarchical fallbacks: Resolve sparse data via (subregion → region → national → overall) strategies.
- Post-process constraints: Apply hospital capacity constraints after time assignment.
- Memory efficiency: Process in large vectorized batches with optional chunking.

## Architecture Overview

1) Vectorized Patient Generation (attributes without date/time)
- Load dynamic patterns via `PatternAnalyzer`:
  - Regional hospital allocation patterns.
  - KTAS hierarchical distributions.
  - Demographics; visit method; chief complaint; department.
  - Treatment results conditional on KTAS and age.
- Generate demographics using observed multi-variate distributions.
- Assign initial hospital using regional choice patterns (and hierarchical fallback if needed).
- Sample independent clinical attributes (visit method, chief complaint, department) from learned distributions.
- Sample KTAS per (region, hospital_type) group using hierarchical distributions (batch sampling).
- Sample treatment outcomes per (KTAS, age_group) batch using learned conditional distributions.

2) Temporal Assignment (date/time)
- Load temporal distributions (monthly, weekday, hourly) from original data.
- Compute daily volumes using multiplicative effects (seasonality, weekday, holiday) with Poisson variability.
- Assign dates by inverse-CDF over daily volumes; assign times using hourly distributions.
- Fallback hourly shape comes from config if hourly patterns are unavailable.

3) Capacity Constraints (post-processing)
- Compute dynamic daily capacity limits per hospital with weekend/holiday multipliers and safety margin.
- Detect overflows where assigned load exceeds daily capacity.
- Reassign overflowed patients prioritizing same-region hospitals with available capacity; fall back to global availability or second-choice probability where configured.
- Record overflow flags and redistribution method.

4) Storage and Reporting
- Persist clinical records and allocation summaries to the target database.
- Generate capacity and performance reports; optional validation reports.

## Caching and Reuse

- All pattern analyses are cached by a data-hash (row count + sampled content) and stored as pickles with JSON metadata.
- Cached results are reused automatically unless the source data changes.

## Validation (Optional Quality Gate)

- Statistical validation compares distributions and correlations between original vs synthetic data.
- Clinical validation checks rule-based consistency (KTAS-outcome, temporal order, medical ranges, etc.).
- A combined score can gate the pipeline to ensure fidelity and clinical plausibility.

## Configurability

- All fallback behavior is driven by configuration (see `config/generation_params.yaml`).
- CLI controls batch sizes, temporal resolution, capacity multipliers, overflow redistribution strategy, and quality gate threshold.

