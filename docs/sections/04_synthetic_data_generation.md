# 4. Synthetic Data Generation

## 4.1 Vectorized Patient Generator

### 4.1.1 Mathematical Foundation of Vectorization

Vectorized generation leverages **Single Instruction, Multiple Data (SIMD)** architecture and **broadcasting** principles to achieve significant performance gains:

**Performance Analysis**: 
- Loop-based generation: $O(n)$ with high constant factor due to Python overhead
- Vectorized generation: $O(n)$ with low constant factor using optimized C implementations
- **Speed improvement**: Typically 10-100x faster for large datasets

**Memory Access Patterns**: Vectorized operations benefit from:
- **Spatial locality**: Consecutive memory access patterns
- **Cache efficiency**: Better L1/L2 cache utilization
- **SIMD utilization**: Parallel execution of identical operations

### 4.1.2 Advanced Vectorized Generation Architecture

The **Vectorized Patient Generator** is the core engine for creating synthetic emergency department patients. It employs sophisticated statistical techniques to ensure both computational efficiency and statistical fidelity.

#### Patient Generation Configuration

The generation system is highly configurable to balance performance with memory constraints:

| Parameter | Default Value | Purpose | Performance Impact |
|-----------|---------------|---------|-------------------|
| **Total Records** | 322,573 | Target dataset size | Linear memory scaling |
| **Batch Size** | 50,000 | Memory-optimized chunks | Cache efficiency |
| **Memory Efficient** | True | Optimize data types | 40-60% memory reduction |
| **Enable Correlation** | True | Preserve statistical relationships | 15% computation overhead |
| **Correlation Method** | Cholesky | Matrix decomposition approach | Most efficient for positive-definite |

#### Memory Optimization Strategy

The system implements several memory optimization techniques:

**Batch Size Optimization**:
1. **Cache Alignment**: Aligns batch sizes to CPU cache line boundaries (64 bytes)
2. **Memory Pages**: Optimizes for system memory page sizes (typically 4KB)
3. **SIMD Width**: Considers SIMD instruction width for vectorization

**Data Type Optimization**:
| Column | Original Type | Optimized Type | Memory Savings |
|--------|---------------|----------------|----------------|
| `pat_age` | int64 (8 bytes) | uint8 (1 byte) | 87.5% reduction |
| `pat_sex` | object | category | ~50% reduction |
| `pat_sarea` | object | category | ~60% reduction |
| `ktas_lv` | int64 (8 bytes) | uint8 (1 byte) | 87.5% reduction |

### 4.1.3 Correlation-Preserving Generation Process

The generation process maintains statistical relationships between variables using advanced multivariate techniques:

#### Multi-Step Generation Algorithm

```
Correlation Matrix → Cholesky Decomposition → Multivariate Normal → Transform to Marginals → Apply Constraints
```

**Step 1: Correlation Structure**
The system builds a correlation matrix based on learned patterns:

| Variable Pair | Correlation | Clinical Reasoning |
|---------------|-------------|-------------------|
| Age ↔ Temporal Factor | 0.15 | Older patients often arrive during day shifts |
| Gender ↔ Region | 0.05 | Slight demographic variations by geography |
| Age ↔ KTAS | Variable | Complex relationship modeled separately |

**Step 2: Multivariate Normal Generation**
- Generate correlated standard normal variables using Cholesky decomposition
- Transform to uniform [0,1] using cumulative distribution function
- Preserve correlation structure throughout transformation

**Step 3: Marginal Distribution Transformation**

Each variable is transformed to match learned distributions:

#### Age Distribution Transformation

**Process**: Inverse Cumulative Distribution Function (CDF) Method

1. **Build Empirical CDF**: From learned age distribution patterns
2. **Interpolation**: Use linear interpolation for smooth transformation
3. **Validation**: Ensure generated ages fall within valid range (0-120)

**Mathematical Foundation**:
$$F^{-1}(u) = \inf\{x : F(x) \geq u\}$$

Where $F^{-1}$ is the quantile function and $u \sim \text{Uniform}[0,1]$.

#### Gender Distribution with Age Conditioning

**Conditional Generation**: $P(\text{Gender} | \text{Age Group})$

| Age Group | Male Probability | Female Probability | Clinical Rationale |
|-----------|------------------|-------------------|-------------------|
| **Infant (0-2)** | 0.52 | 0.48 | Slight male birth rate advantage |
| **Child (2-18)** | 0.51 | 0.49 | Gender-neutral healthcare usage |
| **Adult (18-65)** | 0.48 | 0.52 | Women higher healthcare utilization |
| **Elderly (65+)** | 0.45 | 0.55 | Women longer life expectancy |

#### Regional Distribution with Demographics

**Hierarchical Regional Assignment**:
1. **Major Region Selection**: Based on population-weighted probabilities
2. **Detailed Region**: Conditional on major region selection
3. **Demographic Conditioning**: Slight adjustments based on age/gender patterns

#### KTAS Generation with Multinomial Logistic Regression

**Advanced Conditional Modeling**: The system uses a sophisticated approach to generate KTAS levels based on multiple factors.

**Feature Engineering**:
| Feature | Transformation | Purpose |
|---------|----------------|---------|
| **Age** | age/100.0 | Normalization for numerical stability |
| **Gender** | 1 if Male, 0 if Female | Binary encoding |
| **Region** | region_code/50.0 | Normalized geographic factor |
| **Temporal** | [0,1] continuous | Time-of-day effects |
| **Age²** | (age/100.0)² | Non-linear age effects |
| **Age×Time** | age × temporal | Interaction effects |

**Multinomial Logistic Model**:
$$P(\text{KTAS}=k|\mathbf{x}) = \frac{\exp(\mathbf{x}^T\boldsymbol{\beta}_k)}{1 + \sum_{j=2}^{5}\exp(\mathbf{x}^T\boldsymbol{\beta}_j)}$$

**Clinical Coefficient Interpretation**:
| KTAS Level | Age Effect | Gender Effect | Time Effect | Clinical Meaning |
|------------|------------|---------------|-------------|-----------------|
| **KTAS 2** | +0.3 | -0.1 (F favored) | +0.2 | High acuity increases with age |
| **KTAS 3** | 0.1 | 0.0 (neutral) | 0.0 | Baseline reference |
| **KTAS 4** | -0.2 | +0.1 (M favored) | -0.1 | Lower acuity for younger patients |
| **KTAS 5** | -0.5 | +0.2 (M favored) | -0.3 | Non-urgent skews young, male |

### 4.1.4 Performance Monitoring and Validation

#### Generation Statistics Tracking

The system continuously monitors performance metrics:

| Metric | Purpose | Typical Values |
|--------|---------|----------------|
| **Generation Rate** | Records per second | 50,000-200,000/sec |
| **Batch Time** | Time per batch | 0.5-2.0 seconds |
| **Memory Usage** | Peak memory consumption | 200-500 MB |
| **Cache Hit Rate** | Pattern cache efficiency | >95% |

#### Data Validation Framework

**Multi-Level Validation Process**:

1. **Basic Validation**:
   - Age range: 0-120 years
   - Gender values: Only 'M' or 'F'
   - KTAS range: 1-5 levels
   - No null values

2. **Statistical Validation**:
   - KTAS distribution within clinical ranges
   - Age-gender correlations reasonable
   - Regional distribution matches targets

3. **Clinical Validation**:

**KTAS Distribution Validation**:
| KTAS Level | Expected Range | Clinical Interpretation |
|------------|----------------|------------------------|
| **KTAS 1** | 1-10% | Critical/Resuscitation |
| **KTAS 2** | 10-25% | Emergent/High Urgency |
| **KTAS 3** | 25-45% | Urgent/Moderate |
| **KTAS 4** | 20-40% | Less Urgent/Low |
| **KTAS 5** | 5-20% | Non-Urgent/Very Low |

## 4.2 Temporal Pattern Assignment

### 4.2.1 NHPP (Non-Homogeneous Poisson Process) - Mathematical Foundation

**Theoretical Background**: Emergency department arrivals follow a **Non-Homogeneous Poisson Process** where the intensity function $\lambda(t)$ varies over time:

$$N(t) \sim \text{Poisson}\left(\int_0^t \lambda(s) ds\right)$$

**Key Properties**:
1. **Independence**: Non-overlapping intervals have independent counts
2. **Memoryless**: Future arrivals don't depend on past history
3. **Time-varying intensity**: $\lambda(t)$ captures circadian, weekly, and seasonal patterns

**Intensity Function Modeling**:
$$\lambda(t) = \lambda_0(t) \cdot f_{\text{week}}(t) \cdot f_{\text{season}}(t) \cdot f_{\text{holiday}}(t)$$

Where:
- $\lambda_0(t)$: Base circadian pattern (24-hour cycle)
- $f_{\text{week}}(t)$: Day-of-week effects
- $f_{\text{season}}(t)$: Seasonal variations
- $f_{\text{holiday}}(t)$: Holiday adjustments

### 4.2.2 Advanced Temporal Pattern Implementation

The **Advanced Temporal Pattern Assigner** employs multiple sophisticated modeling techniques to capture the complex temporal dynamics of emergency department arrivals.

#### Multi-Model Temporal Framework

The system uses a hierarchy of temporal models to capture different time scales:

| Model Type | Time Scale | Technique | Purpose |
|------------|------------|-----------|---------|
| **Circadian** | 24-hour cycle | Fourier Series | Daily arrival patterns |
| **Weekly** | 7-day cycle | Harmonic Analysis | Day-of-week effects |
| **Seasonal** | Annual cycle | Decomposition | Flu season, holidays |
| **Holiday** | Event-based | Regression | Special day adjustments |

#### Circadian Pattern Modeling

**Fourier Series Approach**:
The system models daily arrival patterns using Fourier series decomposition:

$$\lambda_{\text{circadian}}(t) = a_0 + \sum_{k=1}^{K} [a_k\cos(2\pi k t/24) + b_k\sin(2\pi k t/24)]$$

**Process Overview**:
1. **Data Aggregation**: Aggregate arrivals to 30-minute intervals
2. **Rate Calculation**: Convert counts to arrivals per hour
3. **Fourier Fitting**: Fit up to 6 harmonics for smooth representation
4. **Validation**: Cross-validate using held-out time periods

**Typical Daily Pattern**:
| Time Period | Relative Intensity | Clinical Insight |
|-------------|-------------------|------------------|
| **Night (00-06)** | 0.3-0.5 | Lowest activity, severe cases |
| **Morning (06-12)** | 0.8-1.2 | Gradual increase, chronic issues |
| **Afternoon (12-18)** | 1.0-1.4 | Peak activity, diverse cases |
| **Evening (18-24)** | 0.7-1.1 | Moderate activity, injuries |

#### Weekly Pattern Analysis

**Harmonic Modeling for Weekly Cycles**:

The system captures day-of-week effects using harmonic analysis:

$$f_{\text{week}}(d) = 1 + \sum_{k=1}^{2} [c_k\cos(2\pi k d/7) + s_k\sin(2\pi k d/7)]$$

Where $d$ is day of week (0-6).

**Weekly Pattern Insights**:
| Day | Multiplier | Pattern Explanation |
|-----|------------|-------------------|
| **Monday** | 1.15 | "Monday Effect" - weekend issues surface |
| **Tuesday** | 1.10 | High activity continues |
| **Wednesday** | 1.00 | Baseline activity |
| **Thursday** | 1.05 | Moderate increase |
| **Friday** | 1.20 | Weekend preparation, late week issues |
| **Saturday** | 0.90 | Lower routine visits |
| **Sunday** | 0.85 | Lowest activity day |

#### Seasonal Decomposition

**Multiplicative Seasonal Model**:
$$Y_t = T_t \times S_t \times I_t \times \epsilon_t$$

Where:
- $T_t$: Trend component (long-term changes)
- $S_t$: Seasonal component (annual cycle)
- $I_t$: Irregular component (random variations)
- $\epsilon_t$: Error term

**Seasonal Implementation Process**:

1. **Trend Estimation**:
   - Use centered moving averages for annual cycle
   - Handle even/odd period adjustments
   - Smooth using Loess regression

2. **Seasonal Component**:
   - Calculate detrended series: $Y_t / T_t$
   - Average over seasonal periods
   - Normalize to mean = 1

3. **Residual Analysis**:
   - Extract irregular component: $Y_t / (T_t \times S_t)$
   - Validate residuals for randomness

**Annual Seasonal Patterns**:
| Season | Multiplier | Medical Drivers |
|--------|------------|----------------|
| **Winter (Dec-Feb)** | 1.25 | Flu season, respiratory issues, falls on ice |
| **Spring (Mar-May)** | 1.05 | Allergies, moderate activity |
| **Summer (Jun-Aug)** | 0.95 | Lower respiratory issues, higher trauma |
| **Fall (Sep-Nov)** | 1.10 | Back-to-school, early flu season |

#### Holiday Effect Modeling

**Regression-Based Holiday Adjustments**:

The system models holiday effects using indicator variables and proximity effects:

**Holiday Feature Engineering**:
| Feature Type | Examples | Effect Window |
|--------------|----------|---------------|
| **Direct Holiday** | Christmas Day, New Year's Day | Single day |
| **Pre-Holiday** | Christmas Eve (-1 day) | 1-3 days before |
| **Post-Holiday** | Day after (+1 day) | 1-3 days after |
| **Holiday Period** | Christmas Week | Extended periods |

**Major Holiday Effects**:
| Holiday | Direct Effect | Pre-Effect | Post-Effect | Pattern |
|---------|---------------|------------|-------------|---------|
| **Christmas** | 0.3× | 0.7× | 1.5× | Very low, then surge |
| **New Year** | 0.4× | 0.8× | 1.3× | Low, moderate recovery |
| **Thanksgiving** | 0.6× | 0.9× | 1.2× | Moderate decrease |
| **July 4th** | 0.8× | 1.1× | 1.1× | Slight increase (trauma) |

#### NHPP Arrival Generation

**Thinning Algorithm Implementation**:

For generating actual arrival times, the system uses the thinning (rejection) algorithm:

**Algorithm Steps**:
1. **Upper Bound**: Find maximum intensity $\lambda_{\max} = \max_t \lambda(t)$
2. **Homogeneous Generation**: Generate arrivals using rate $\lambda_{\max}$
3. **Acceptance Test**: Accept each arrival with probability $\lambda(t)/\lambda_{\max}$
4. **Rejection**: Reject arrivals that fail the test

**Efficiency Optimization**:
- **Adaptive Upper Bounds**: Use time-varying upper bounds
- **Batch Processing**: Generate multiple arrivals simultaneously
- **Vectorized Operations**: Optimize acceptance testing

**Validation Process**:
| Test Type | Method | Acceptance Criteria |
|-----------|--------|-------------------|
| **Rate Validation** | Compare generated vs. target rates | <5% deviation |
| **Goodness-of-Fit** | Kolmogorov-Smirnov test | p-value > 0.05 |
| **Autocorrelation** | Check for independence | No significant correlation |
| **Seasonal Alignment** | Validate seasonal patterns | Match expected multipliers |

## 4.3 Hospital Allocation System

### 4.3.1 Regional-Based Allocation Strategy

The hospital allocation system replaces traditional gravity models with a **data-driven regional approach** that learns actual patient flow patterns from historical data.

#### Pattern-Based Hospital Selection

**Hierarchical Selection Process**:
```
Patient Region → Regional Hospital Pools → Capacity-Weighted Selection → Final Assignment
```

**Regional Hospital Grouping**:
| Region Code | Primary Hospitals | Secondary Hospitals | Tertiary Hospitals |
|-------------|------------------|-------------------|-------------------|
| **11 (Seoul)** | 15 major centers | 45 regional hospitals | 8 specialty centers |
| **21 (Busan)** | 8 major centers | 25 regional hospitals | 4 specialty centers |
| **Other Regions** | 2-5 major centers | 10-20 regional hospitals | 1-2 specialty centers |

#### Dynamic Capacity Management

**Real-Time Capacity Factors**:
| Factor | Weight | Update Frequency | Impact |
|--------|--------|------------------|--------|
| **Base Capacity** | 0.4 | Static | Fundamental hospital size |
| **KTAS Specialization** | 0.3 | Monthly | Specialized care capabilities |
| **Regional Preference** | 0.2 | Annually | Patient flow preferences |
| **Time-of-Day Adjustment** | 0.1 | Hourly | Shift-based capacity |

## 4.4 Clinical Feature Generation

### 4.4.1 Vital Signs Generation

The system generates realistic vital signs using **multivariate Gaussian mixture models** that capture complex physiological relationships.

#### Correlation Structure

**Key Physiological Relationships**:
| Variable Pair | Correlation | Physiological Basis |
|---------------|-------------|-------------------|
| **Systolic BP ↔ Diastolic BP** | 0.85 | Direct cardiovascular relationship |
| **Heart Rate ↔ Respiratory Rate** | 0.35 | Cardiorespiratory coupling |
| **Age ↔ Blood Pressure** | 0.45 | Age-related vascular changes |
| **KTAS ↔ Heart Rate** | -0.25 | Stress response in acute conditions |

#### Age-Stratified Normal Ranges

| Age Group | Systolic BP | Diastolic BP | Heart Rate | Resp Rate |
|-----------|-------------|--------------|------------|-----------|
| **Pediatric (0-17)** | 95-115 | 55-75 | 80-120 | 16-24 |
| **Adult (18-64)** | 110-140 | 70-90 | 60-100 | 12-20 |
| **Elderly (65+)** | 120-160 | 75-95 | 65-105 | 14-22 |

### 4.4.2 Diagnosis Code Generation

**Clinical Syndrome Modeling**: Uses Latent Dirichlet Allocation (LDA) to capture diagnostic patterns.

**Common Emergency Syndromes**:
| Syndrome | ICD-10 Codes | KTAS Distribution | Prevalence |
|----------|-------------|------------------|------------|
| **Chest Pain** | R06.02, I20.9 | KTAS 2-3 | 15% |
| **Abdominal Pain** | R10.9, K59.9 | KTAS 3-4 | 12% |
| **Respiratory** | J44.0, J18.9 | KTAS 2-4 | 10% |
| **Trauma** | S72.9, S06.9 | KTAS 1-3 | 8% |

---