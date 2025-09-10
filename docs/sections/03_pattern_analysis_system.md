# 3. Pattern Analysis System

## 3.1 Dynamic Pattern Learning

The pattern analysis system employs sophisticated statistical learning techniques to extract distributions and relationships directly from source data, avoiding hardcoded assumptions that may not reflect real-world variations.

### 3.1.1 Theoretical Foundation

**Statistical Learning Principle**: Instead of assuming parametric distributions (e.g., normal, exponential), the system uses empirical distribution functions and kernel density estimation to capture the true data-generating process:

$$\hat{F}_n(x) = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}(X_i \leq x)$$

Where $\hat{F}_n(x)$ is the empirical distribution function, and $\mathbb{1}$ is the indicator function.

**Adaptive Smoothing**: For continuous variables, kernel density estimation provides smooth probability density functions:

$$\hat{f}(x) = \frac{1}{nh}\sum_{i=1}^{n} K\left(\frac{x-X_i}{h}\right)$$

Where $K$ is the kernel function (typically Gaussian) and $h$ is the bandwidth selected via cross-validation.

### 3.1.2 Pattern Categories and Algorithmic Approaches

**Demographic Patterns**

*Age Distribution Learning*:
- **Algorithm**: Adaptive binning with optimal bin selection using Scott's rule: $h = 3.5\sigma n^{-1/3}$
- **Reasoning**: Age distributions are often multimodal (pediatric, adult, geriatric peaks) requiring flexible binning
- **Stratification**: Conditional distributions $P(\text{age}|\text{region}, \text{time})$ to capture regional and temporal variations

*Gender Ratio Estimation*:
- **Algorithm**: Bayesian beta-binomial modeling with conjugate priors
- **Mathematical Model**: 
  $$P(\text{male}|\alpha, \beta) \sim \text{Beta}(\alpha, \beta)$$
  $$\text{Observations} \sim \text{Binomial}(n, P(\text{male}))$$
- **Reasoning**: Beta-binomial captures uncertainty in gender ratios while providing conjugate updates

*Geographic Clustering*:
- **Algorithm**: Hierarchical spatial clustering using Ward linkage
- **Distance Metric**: Haversine distance for geographic coordinates
- **Reasoning**: Emergency care patterns exhibit spatial autocorrelation due to population density and healthcare accessibility

**Temporal Patterns**

*Non-Homogeneous Poisson Process (NHPP) Learning*:
- **Theoretical Model**: Arrival rate $\lambda(t)$ varies over time
  $$N(t) \sim \text{Poisson}\left(\int_0^t \lambda(s)ds\right)$$
- **Algorithm**: 
  1. Estimate intensity function using kernel regression: $\hat{\lambda}(t) = \sum_{i} K_h(t - t_i)$
  2. Smooth using Gaussian kernels with adaptive bandwidth
  3. Validate goodness-of-fit using Kolmogorov-Smirnov test
- **Reasoning**: Hospital arrivals follow time-varying patterns (rush hours, shift changes, circadian rhythms)

*Seasonal Decomposition*:
- **Algorithm**: X-13ARIMA-SEATS seasonal adjustment
- **Mathematical Framework**:
  $$Y_t = T_t + S_t + I_t + \epsilon_t$$
  Where $T_t$ is trend, $S_t$ is seasonal, $I_t$ is irregular, $\epsilon_t$ is noise
- **Reasoning**: Healthcare utilization exhibits strong seasonal patterns (flu season, holidays, weather effects)

*Day-of-Week and Holiday Effects*:
- **Algorithm**: Fourier series decomposition with harmonic analysis
- **Mathematical Model**:
  $$\lambda(t) = \mu + \sum_{k=1}^{K} [a_k\cos(2\pi k t/T) + b_k\sin(2\pi k t/T)]$$
- **Reasoning**: Weekly patterns are periodic; Fourier analysis captures multiple harmonics

**Clinical Patterns**

*KTAS Distribution Learning*:
- **Algorithm**: Multinomial logistic regression with demographic predictors
- **Mathematical Model**:
  $$P(\text{KTAS}=k|\mathbf{x}) = \frac{\exp(\mathbf{x}^T\boldsymbol{\beta}_k)}{1 + \sum_{j=1}^{4}\exp(\mathbf{x}^T\boldsymbol{\beta}_j)}$$
- **Reasoning**: KTAS severity depends on age, comorbidities, presentation time - logistic regression captures these relationships

*Vital Sign Correlation Learning*:
- **Algorithm**: Multivariate Gaussian mixture modeling with EM algorithm
- **Mathematical Framework**:
  $$\mathbf{V} \sim \sum_{k=1}^{K} \pi_k \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$
- **Reasoning**: Vital signs exhibit complex correlations (BP-HR relationship, respiratory compensation) requiring mixture models

*Diagnosis Pattern Mining*:
- **Algorithm**: Latent Dirichlet Allocation (LDA) for topic modeling
- **Mathematical Model**:
  $$P(\text{diagnosis}|\text{symptoms}) = \sum_{k} P(\text{diagnosis}|\text{topic}_k)P(\text{topic}_k|\text{symptoms})$$
- **Reasoning**: Emergency presentations involve latent clinical syndromes best captured by topic models

### 3.1.3 Advanced Learning Algorithm Implementation

The **Advanced Pattern Analyzer** is the core component responsible for learning statistical patterns from emergency department data. Instead of hardcoded distributions, it dynamically adapts to the actual data characteristics through several intelligent processes:

#### Data Distribution Learning Process

**Automatic Model Selection**: The system intelligently chooses the best statistical model for each data type:

| Data Characteristic | Statistical Test Applied | Model Selected | Reasoning |
|-------------------|------------------------|----------------|-----------|
| **Continuous & Normal** | Shapiro-Wilk, Jarque-Bera | Gaussian Distribution | Parametric efficiency |
| **Continuous & Non-Normal** | Anderson-Darling | Kernel Density Estimation | Flexible non-parametric |
| **Discrete & Count Data** | Dispersion Test | Poisson/Negative Binomial | Count process modeling |
| **Categorical** | Chi-square Goodness-of-Fit | Multinomial with Dirichlet | Bayesian uncertainty |

#### Kernel Density Estimation (KDE) Workflow

When data doesn't follow standard parametric distributions, the system employs a sophisticated KDE process:

1. **Initial Bandwidth Selection**: Uses Scott's rule $h = \sigma \cdot n^{-1/5}$ as starting point
2. **Cross-Validation Refinement**: Tests 20 bandwidth values across logarithmic scale
3. **Optimal Selection**: Chooses bandwidth maximizing likelihood via 5-fold cross-validation
4. **Model Validation**: Ensures the resulting model captures data characteristics accurately

#### Non-Homogeneous Poisson Process (NHPP) Learning

For temporal arrival patterns, the system implements a comprehensive NHPP learning algorithm:

**Process Overview**:
```
Raw Timestamps → Time Conversion → Intensity Estimation → Fourier Smoothing → Validated Model
```

**Detailed Steps**:

1. **Time Normalization**: Convert timestamps to continuous time scale (0-24 hours)
2. **Intensity Estimation**: 
   - Use Gaussian kernels with 2-hour bandwidth
   - Calculate arrival rates at 30-minute intervals
   - Apply kernel density estimation to smooth intensity function
3. **Fourier Series Fitting**:
   - Fit up to 6 harmonics to capture daily patterns
   - Use regularized least squares to prevent overfitting
   - Balance smoothness with pattern fidelity
4. **Model Reconstruction**: Create continuous intensity function for simulation

#### Bootstrap Confidence Intervals

The system quantifies uncertainty in learned patterns using bootstrap methodology:

**Bootstrap Process**:
1. Generate synthetic samples from observed distribution
2. Create 1000 bootstrap resamples with replacement
3. Calculate distribution for each resample
4. Compute 95% confidence intervals using percentiles
5. Validate interval widths (reject if >30% width)

### 3.1.4 Statistical Validation and Model Selection

**Cross-Validation Framework**:
- **Time Series CV**: For temporal patterns, use time-aware splits to avoid data leakage
- **Stratified CV**: For categorical outcomes, maintain class balance across folds
- **Geographic CV**: For spatial patterns, use spatial block cross-validation

**Goodness-of-Fit Testing**:
- **Kolmogorov-Smirnov**: For continuous distributions
- **Chi-square**: For discrete/categorical distributions  
- **Anderson-Darling**: More sensitive to tail behavior than KS test
- **Cramér-von Mises**: For overall distribution shape

**Model Selection Criteria**:
- **AIC/BIC**: Penalized likelihood for parametric models
- **Cross-validation error**: For non-parametric models
- **Domain knowledge**: Clinical constraints on parameter ranges

## 3.2 Hierarchical Fallback Strategy

### 3.2.1 Mathematical Foundation

The hierarchical fallback strategy addresses the **curse of dimensionality** in conditional probability estimation. As we increase conditioning variables, the effective sample size decreases exponentially:

$$n_{\text{effective}} = \frac{n}{k^d}$$

Where $n$ is total sample size, $k$ is average categories per dimension, and $d$ is number of dimensions.

**Statistical Reliability Threshold**: We require minimum sample size $n_{\min}$ for reliable estimation based on the **Central Limit Theorem**:

$$n_{\min} = \left(\frac{z_{\alpha/2} \cdot \sigma}{E}\right)^2$$

Where $z_{\alpha/2}$ is critical z-value (typically 1.96 for 95% confidence), $\sigma$ is estimated standard deviation, and $E$ is acceptable margin of error.

### 3.2.2 Hierarchy Design Principles

**Information-Theoretic Justification**: Each fallback level represents a trade-off between specificity and reliability:

1. **Level 1 (Most Specific)**: $P(Y|X_1, X_2, X_3)$ - Maximum conditional information
2. **Level 2 (Regional)**: $P(Y|X_1, X_3)$ - Remove least informative variable
3. **Level 3 (Temporal)**: $P(Y|X_3)$ - Keep most universal patterns
4. **Level 4 (Global)**: $P(Y)$ - Marginal distribution as ultimate fallback

**Variable Selection Algorithm**: Use **Mutual Information** to rank conditioning variables:

$$I(Y; X_i) = \sum_{x_i, y} p(x_i, y) \log\frac{p(x_i, y)}{p(x_i)p(y)}$$

Higher mutual information variables are retained in fallback levels.

### 3.2.3 Hierarchical Pattern Management System

The **Hierarchical Pattern Manager** implements a sophisticated fallback system that ensures reliable pattern retrieval even when specific conditions have insufficient data.

#### Hierarchy Structure Design

The system organizes patterns across 4 hierarchical levels, each representing different levels of specificity:

| Level | KTAS Distribution | Hospital Allocation | Temporal Patterns | Sample Requirements |
|-------|------------------|-------------------|------------------|-------------------|
| **1** | Region(4-digit) + Hospital Type + Time Period | Region(4-digit) + Age Group + Severity | Region(2-digit) + Hospital Size + Day Type | ≥30 samples |
| **2** | Region(2-digit) + Hospital Type + Time Period | Region(2-digit) + Age Group + Severity | Region(2-digit) + Day Type | ≥20 samples |
| **3** | Hospital Type + Time Period | Age Group + Severity | Day Type only | ≥15 samples |
| **4** | Marginal Distribution | Marginal Distribution | Marginal Distribution | Always available |

#### Pattern Retrieval Algorithm

The system follows a systematic approach to pattern retrieval:

```
Pattern Request → Level 1 Attempt → Statistical Validation → Success/Failure
                     ↓ (if failure)
                 Level 2 Attempt → Statistical Validation → Success/Failure
                     ↓ (if failure)
                 Level 3 Attempt → Statistical Validation → Success/Failure
                     ↓ (if failure)
                 Level 4 (Guaranteed) → Return Global Pattern
```

#### Statistical Validation Framework

Each pattern undergoes rigorous validation before acceptance:

**Validation Criteria**:

1. **Sample Size Test**: Minimum 30 samples for reliable statistics
2. **Bootstrap Stability**: Confidence intervals must be <30% width
3. **Entropy Check**: Information content ≥1 bit (log₂(2))
4. **Variance Check**: Avoid degenerate distributions (variance >10⁻⁶)

**Quality Assessment Process**:

| Criterion | Test Method | Acceptance Threshold | Reasoning |
|-----------|-------------|---------------------|-----------|
| **Sample Size** | Simple Count | n ≥ 30 | Central Limit Theorem |
| **Stability** | Bootstrap CI Width | Width < 30% | Statistical Precision |
| **Information** | Shannon Entropy | H ≥ 1 bit | Meaningful Patterns |
| **Variance** | Distribution Spread | σ² > 10⁻⁶ | Non-degenerate |

#### Uncertainty Quantification

The system provides uncertainty measures based on hierarchy level:

**Uncertainty Multipliers**:
- **Level 1**: 1.0× (Most specific, lowest uncertainty)
- **Level 2**: 1.2× (Regional aggregation)
- **Level 3**: 1.5× (National patterns)
- **Level 4**: 2.0× (Global fallback, highest uncertainty)

**Pattern Enhancement**: Each returned pattern includes:
- **Hierarchy Level**: Which level was used
- **Uncertainty Factor**: Relative confidence measure
- **Reliability Score**: Statistical quality indicator
- **Confidence Intervals**: Bootstrap-derived uncertainty bounds

#### Pattern Quality Metrics

The system provides comprehensive quality assessment:

**Reliability Score Calculation**:
1. **Sample Size Factor**: min(1.0, sample_size / (2 × min_required))
2. **Entropy Factor**: observed_entropy / maximum_possible_entropy
3. **Variance Factor**: 1 - |variance - optimal_variance| / optimal_variance
4. **Overall Score**: Mean of all factors

**Information Content**: Shannon entropy of the distribution
**Statistical Power**: Ability to detect true effects (Cohen's conventions)

## 3.3 Pattern Caching Mechanism

### 3.3.1 Cache Structure

The caching system uses a structured approach to store learned patterns:

```
CachedPattern:
├── pattern_type: string (e.g., "ktas_distribution")
├── data_hash: string (MD5 of source data)
├── timestamp: datetime (when pattern was learned)
├── pattern_data: dictionary (actual pattern values)
└── metadata: dictionary (quality metrics, sample sizes)
```

### 3.3.2 Cache Management

- **Cache Key**: Combination of table name, pattern type, and data hash
- **Invalidation**: Automatic when source data changes
- **Storage**: JSON files in `cache/patterns/` directory
- **Performance**: 100x speedup for repeated analyses

---