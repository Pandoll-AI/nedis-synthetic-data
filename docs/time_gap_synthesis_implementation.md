# Time Gap Synthesis Implementation Plan

## Overview

This document describes the implementation of severity-adjusted time gap synthesis for the NEDIS synthetic data generation system. The approach follows the core principle of **dynamic pattern learning** from actual data, avoiding any hardcoded distributions.

## Time-Dependent Variables Identified

### Date-Time Pairs in NEDIS Dataset
1. **Visit DateTime** (`vst_dt` + `vst_tm`): ER arrival time
2. **Occurrence DateTime** (`ocur_dt` + `ocur_tm`): Incident/symptom onset time  
3. **ER Discharge DateTime** (`otrm_dt` + `otrm_tm`): ER departure time
4. **Admission DateTime** (`inpat_dt` + `inpat_tm`): Hospital admission time
5. **Outpatient DateTime** (`otpat_dt` + `otpat_tm`): Outpatient referral time

### DateTime Format Conversion
- Original format: `YYMMDD` (date) + `HHMM` (time)
- Example: `170123` + `0350` → `2017-01-23 03:50:00`
- Conversion function implemented in `TimeGapSynthesizer._parse_datetime()`

## Key Time Gaps and KTAS Correlations

### Analysis Results Summary

Based on analysis of 100,000 records from `nedis_data.nedis2017`:

| KTAS Level | Sample Size | Avg ER Stay (min) | Description |
|------------|-------------|-------------------|-------------|
| 1 (Critical) | 657 | Data-driven | Shortest stays, immediate care |
| 2 (Emergency) | 3,472 | Data-driven | Very urgent, quick processing |
| 3 (Urgent) | 31,210 | Data-driven | Moderate urgency |
| 4 (Less Urgent) | 55,579 | Data-driven | Longer waits acceptable |
| 5 (Non-urgent) | 9,082 | Data-driven | Routine care |

### Time Gap Types

1. **ER Stay Duration** (`gap_vst_to_otrm`)
   - Time from ER arrival to ER discharge
   - Strongly correlated with KTAS level
   - Critical patients (KTAS 1) have shorter but intensive stays
   - Non-urgent patients (KTAS 5) may have variable stays

2. **Admission Decision Time** (`gap_vst_to_inpat`)
   - Time from ER arrival to hospital admission
   - Applicable only for admitted patients (treatment results 31-34)
   - Faster decisions for critical patients

3. **Incident to Arrival** (`gap_ocur_to_vst`)
   - Time from symptom onset to ER arrival
   - Reflects urgency of condition and transport time

## Implementation Architecture

### 1. TimeGapSynthesizer Class
Location: `src/temporal/time_gap_synthesizer.py`

```python
class TimeGapSynthesizer:
    """
    Generates time gaps based on:
    - KTAS severity levels
    - Treatment outcomes
    - Learned distributions from actual data
    """
    
    def analyze_time_patterns()
        # Dynamically learn patterns from original data
        
    def generate_time_gaps(ktas_levels, treatment_results, visit_datetimes)
        # Generate time gaps using hierarchical distributions
```

### 2. Hierarchical Distribution Strategy

```
Level 1: KTAS + Treatment Result
    ↓ (fallback if insufficient data)
Level 2: KTAS Only  
    ↓ (fallback if insufficient data)
Level 3: Overall Average
    ↓ (fallback if no data)
Level 4: Clinical Defaults (safety net only)
```

### 3. Distribution Modeling

- **Primary Model**: Log-normal distribution
  - Natural fit for time duration data (always positive, right-skewed)
  - Parameters: shape, location, scale
  - Fitted using `scipy.stats.lognorm`

- **Constraints Applied**:
  - Minimum ER stay by KTAS (clinical safety)
  - Maximum reasonable stay (data quality)
  - Logical ordering (arrival < discharge < admission)

## Integration with Existing Pipeline

### Modified VectorizedPatientGenerator Flow

```python
# In VectorizedPatientGenerator.generate_all_patients()

# Step 1: Generate base patient data (existing)
patients_df = self._generate_patients_vectorized(total_records)

# Step 2: Assign temporal patterns (existing)
temporal_assigner = TemporalPatternAssigner(self.db, self.config)
patients_df = temporal_assigner.assign_temporal_patterns(patients_df)

# Step 3: NEW - Generate time gaps based on KTAS
time_synthesizer = TimeGapSynthesizer(self.db, self.config)
time_gaps_df = time_synthesizer.generate_time_gaps(
    ktas_levels=patients_df['ktas01'],
    treatment_results=patients_df['emtrt_rust'],
    visit_datetimes=patients_df['vst_datetime']
)

# Step 4: Merge time gaps into patient data
patients_df = pd.concat([patients_df, time_gaps_df], axis=1)
```

## Clinical Validation Rules

### Temporal Logic Constraints
1. `ocur_datetime ≤ vst_datetime` (incident before arrival)
2. `vst_datetime < otrm_datetime` (arrival before discharge)
3. `vst_datetime < inpat_datetime` (arrival before admission)
4. `inpat_datetime ≤ otrm_datetime` (admission before/at ER discharge)

### KTAS-Based Constraints
- KTAS 1: Minimum 15 min, Maximum 12 hours
- KTAS 2: Minimum 30 min, Maximum 24 hours
- KTAS 3: Minimum 45 min, Maximum 48 hours
- KTAS 4: Minimum 60 min, Maximum 48 hours
- KTAS 5: Minimum 30 min, Maximum 24 hours

## Usage Example

```python
from src.temporal.time_gap_synthesizer import TimeGapSynthesizer
from src.core.database import DatabaseManager
from src.core.config import ConfigManager

# Initialize
db = DatabaseManager('nedis_data.duckdb')
config = ConfigManager()
synthesizer = TimeGapSynthesizer(db, config)

# Analyze patterns (cached after first run)
patterns = synthesizer.analyze_time_patterns()

# Generate time gaps for synthetic patients
time_gaps_df = synthesizer.generate_time_gaps(
    ktas_levels=np.array([1, 2, 3, 4, 5]),
    treatment_results=np.array(['31', '11', '32', '11', '11']),
    visit_datetimes=pd.Series([datetime.now()] * 5)
)
```

## Benefits of This Approach

1. **No Hardcoding**: All distributions learned from actual data
2. **Severity-Adjusted**: Respects clinical urgency patterns
3. **Hierarchical Fallback**: Robust to data sparsity
4. **Clinically Valid**: Enforces medical logic constraints
5. **Performance**: Vectorized operations for efficiency
6. **Cacheable**: Patterns cached for reuse

## Testing Strategy

### Unit Tests
- Distribution fitting accuracy
- Hierarchical fallback logic
- Constraint enforcement
- DateTime parsing

### Integration Tests
- Full pipeline with time gap generation
- Clinical rule validation
- Performance benchmarks

### Validation Metrics
- KS test for distribution similarity
- Clinical rule compliance rate
- Temporal logic consistency

## Next Steps

1. ✅ Implement TimeGapSynthesizer class
2. ⏳ Integrate with VectorizedPatientGenerator
3. ⏳ Add unit tests for time gap generation
4. ⏳ Validate against original data distributions
5. ⏳ Performance optimization if needed

## References

- Original analysis script: `scripts/analyze_time_gaps.py`
- Analysis results: `outputs/time_gap_analysis.json`
- Synthesis plan: `outputs/time_gap_synthesis_plan.json`
- Implementation: `src/temporal/time_gap_synthesizer.py`