# EnbPI Replication Study — Summary Report

## 1. Overview

**Paper:** Xu & Xie, "Conformal Prediction Interval for Dynamic Time-Series" (ICML 2021)

**Purpose:** Replicate the paper's experiments using (1) the original GitHub implementation and (2) MAPIE's TimeSeriesRegressor, then compare results and verify whether each implementation matches the paper's Algorithm 1.

**Dataset:** Solar Atlanta (DHI response, 8760 observations, capped at 10,000)

## 2. Methodology

### Implementations Compared
| Implementation | Description |
|---|---|
| **Original GitHub** | `PI_class_EnbPI.py` from github.com/hamrel-cxu/EnbPI, imported and used directly |
| **Reimplementation** | Standalone re-implementation of Algorithm 1 for cross-validation (trial 0 only) |
| **MAPIE Online** | `mapie.regression.TimeSeriesRegressor` with `method='enbpi'`, online score updates |
| **MAPIE Batch** | Same MAPIE model but without score updates (uses only training residuals) |

### Regressors
| Regressor | Configuration |
|---|---|
| Ridge | `RidgeCV(alphas=linspace(0.0001, 10, 10))` |
| RF | `RandomForestRegressor(n_estimators=10, max_depth=2)` |
| MLP | `MLPRegressor(hidden_layer_sizes=(20,), max_iter=50, early_stopping=True)` — proxy for paper's Keras NeuralNet |

### Experimental Settings
- **Section 5.1:** alpha in {0.05, 0.10, 0.15, 0.20, 0.25}, train=20%, B=30 bootstrap models, 10 trials
- **Section 5.2:** alpha=0.1 fixed, train sizes from 10-30% (3 settings), B=30, 10 trials
- **Feature types:** Multivariate (original features) and Univariate (lag-20 features)

## 3. Code Inspection: Algorithm 1 vs Implementations

| # | Aspect | Original GitHub vs Paper | MAPIE vs Paper |
|---|---|---|---|
| 1 | Bootstrap method | **MATCHES** (i.i.d.) | **DIFFERS** (block bootstrap) |
| 2 | Point prediction | **MATCHES** ((1-alpha) quantile) | **DIFFERS** (mean aggregation) |
| 3 | Residual sign | **MATCHES** (absolute) | **DIFFERS** (signed, asymmetric) |
| 4 | Sliding window | **MATCHES** | **MATCHES** (functionally) |
| 5 | Model refitting | **MATCHES** (none) | **MATCHES** (none) |
| 6 | Quantile method | **MATCHES** (floor index) | **DIFFERS** (ceiling, conservative) |

**The Original GitHub faithfully implements Algorithm 1.** MAPIE implements an EnbPI-*inspired* method with three meaningful algorithmic differences.

### Re-implementation Validation (Trial 0)
Max coverage difference: 0.000000, max width difference: 0.0000

## 4. Section 5.1 Results — Interval Validity (varying alpha)

### Multivariate Features
| regressor | alpha | Original_GitHub Cov | Original_GitHub Wid | MAPIE_Online Cov | MAPIE_Online Wid | MAPIE_Batch Cov | MAPIE_Batch Wid |
|---|---|---|---|---|---|---|---|
| MLP | 0.05 | 0.954±0.000 | 583.6±1.3 | 0.939±0.000 | 380.5±0.0 | 0.899±0.000 | 265.8±0.0 |
| MLP | 0.10 | 0.900±0.000 | 425.3±1.2 | 0.879±0.000 | 332.3±0.0 | 0.855±0.000 | 199.0±0.0 |
| MLP | 0.15 | 0.853±0.000 | 260.9±0.7 | 0.828±0.000 | 291.0±0.0 | 0.648±0.000 | 147.3±0.0 |
| MLP | 0.20 | 0.801±0.001 | 131.5±0.3 | 0.778±0.000 | 251.1±0.0 | 0.591±0.000 | 112.9±0.0 |
| MLP | 0.25 | 0.753±0.001 | 86.1±0.5 | 0.731±0.000 | 208.0±0.0 | 0.525±0.000 | 84.6±0.0 |
| RF | 0.05 | 0.954±0.000 | 402.5±1.7 | 0.951±0.000 | 369.0±0.0 | 0.873±0.000 | 211.6±0.0 |
| RF | 0.10 | 0.902±0.000 | 283.6±0.8 | 0.905±0.000 | 272.2±0.0 | 0.580±0.000 | 133.1±0.0 |
| RF | 0.15 | 0.854±0.001 | 203.7±1.1 | 0.868±0.000 | 220.1±0.0 | 0.532±0.000 | 96.3±0.0 |
| RF | 0.20 | 0.796±0.002 | 149.6±1.9 | 0.825±0.000 | 180.7±0.0 | 0.465±0.000 | 72.1±0.0 |
| RF | 0.25 | 0.752±0.004 | 127.3±4.1 | 0.788±0.000 | 145.1±0.0 | 0.402±0.000 | 49.5±0.0 |
| Ridge | 0.05 | 0.954±0.000 | 398.0±1.7 | 0.944±0.000 | 352.8±0.0 | 0.801±0.000 | 276.4±0.0 |
| Ridge | 0.10 | 0.904±0.000 | 291.6±0.8 | 0.890±0.000 | 298.6±0.0 | 0.722±0.000 | 213.1±0.0 |
| Ridge | 0.15 | 0.850±0.000 | 217.3±0.6 | 0.836±0.000 | 256.2±0.0 | 0.640±0.000 | 157.0±0.0 |
| Ridge | 0.20 | 0.794±0.001 | 179.8±1.0 | 0.778±0.000 | 214.4±0.0 | 0.568±0.000 | 123.5±0.0 |
| Ridge | 0.25 | 0.743±0.001 | 157.6±1.1 | 0.719±0.000 | 175.1±0.0 | 0.523±0.000 | 105.0±0.0 |

### Univariate Features (lag-20)
| regressor | alpha | Original_GitHub Cov | Original_GitHub Wid | MAPIE_Online Cov | MAPIE_Online Wid | MAPIE_Batch Cov | MAPIE_Batch Wid |
|---|---|---|---|---|---|---|---|
| MLP | 0.05 | 0.952±0.000 | 289.3±0.4 | 0.952±0.000 | 290.4±0.0 | 0.877±0.000 | 166.9±0.0 |
| MLP | 0.10 | 0.902±0.001 | 196.1±0.8 | 0.901±0.000 | 196.5±0.0 | 0.772±0.000 | 107.6±0.0 |
| MLP | 0.15 | 0.849±0.001 | 146.5±0.6 | 0.848±0.000 | 147.8±0.0 | 0.685±0.000 | 78.8±0.0 |
| MLP | 0.20 | 0.795±0.001 | 116.9±0.4 | 0.794±0.000 | 118.0±0.0 | 0.602±0.000 | 59.9±0.0 |
| MLP | 0.25 | 0.748±0.001 | 97.5±0.4 | 0.746±0.000 | 98.5±0.0 | 0.548±0.000 | 48.7±0.0 |
| RF | 0.05 | 0.951±0.001 | 373.9±6.0 | 0.947±0.000 | 310.5±0.0 | 0.879±0.000 | 191.1±0.0 |
| RF | 0.10 | 0.902±0.001 | 256.6±2.7 | 0.902±0.000 | 255.0±0.0 | 0.795±0.000 | 120.4±0.0 |
| RF | 0.15 | 0.852±0.001 | 169.9±0.9 | 0.853±0.000 | 209.1±0.0 | 0.746±0.000 | 89.8±0.0 |
| RF | 0.20 | 0.800±0.001 | 125.4±1.5 | 0.802±0.000 | 161.3±0.0 | 0.702±0.000 | 67.4±0.0 |
| RF | 0.25 | 0.751±0.001 | 95.6±1.2 | 0.756±0.000 | 118.7±0.0 | 0.670±0.000 | 51.9±0.0 |
| Ridge | 0.05 | 0.949±0.000 | 301.0±0.6 | 0.952±0.000 | 301.1±0.0 | 0.877±0.000 | 149.2±0.0 |
| Ridge | 0.10 | 0.899±0.001 | 197.2±1.3 | 0.898±0.000 | 175.9±0.0 | 0.816±0.000 | 100.1±0.0 |
| Ridge | 0.15 | 0.846±0.000 | 139.1±1.2 | 0.846±0.000 | 127.7±0.0 | 0.692±0.000 | 67.0±0.0 |
| Ridge | 0.20 | 0.797±0.001 | 102.8±0.9 | 0.793±0.000 | 99.2±0.0 | 0.584±0.000 | 50.0±0.0 |
| Ridge | 0.25 | 0.744±0.001 | 81.1±0.6 | 0.739±0.000 | 78.8±0.0 | 0.518±0.000 | 42.1±0.0 |

### Plots
- `figure1_replication_multivariate.png` — Coverage and width vs 1-alpha (multivariate)
- `figure1_replication_univariate.png` — Coverage and width vs 1-alpha (univariate)

## 5. Section 5.2 Results — Effect of Training Size (alpha=0.1)

### Multivariate Features
| regressor | train_size | Original_GitHub Cov | Original_GitHub Wid | MAPIE_Online Cov | MAPIE_Online Wid | MAPIE_Batch Cov | MAPIE_Batch Wid |
|---|---|---|---|---|---|---|---|
| MLP | 876 | 0.894±0.000 | 427.2±0.6 | 0.892±0.000 | 318.1±0.0 | 0.855±0.000 | 168.8±0.0 |
| MLP | 1654 | 0.900±0.000 | 420.2±0.9 | 0.883±0.000 | 329.9±0.0 | 0.859±0.000 | 203.8±0.0 |
| MLP | 2433 | 0.915±0.000 | 397.7±0.8 | 0.906±0.000 | 335.7±0.0 | 0.854±0.000 | 245.2±0.0 |
| RF | 876 | 0.897±0.001 | 276.3±4.4 | 0.901±0.000 | 269.7±0.0 | 0.584±0.000 | 136.8±0.0 |
| RF | 1654 | 0.903±0.000 | 281.7±0.8 | 0.907±0.000 | 271.6±0.0 | 0.585±0.000 | 131.6±0.0 |
| RF | 2433 | 0.913±0.000 | 293.1±0.5 | 0.912±0.000 | 291.6±0.0 | 0.603±0.000 | 183.5±0.0 |
| Ridge | 876 | 0.895±0.000 | 271.8±0.9 | 0.890±0.000 | 288.3±0.0 | 0.585±0.000 | 253.8±0.0 |
| Ridge | 1654 | 0.904±0.000 | 289.9±0.7 | 0.893±0.000 | 298.1±0.0 | 0.744±0.000 | 220.5±0.0 |
| Ridge | 2433 | 0.909±0.001 | 298.0±0.4 | 0.883±0.000 | 301.8±0.0 | 0.711±0.000 | 213.3±0.0 |

### Univariate Features (lag-20)
| regressor | train_size | Original_GitHub Cov | Original_GitHub Wid | MAPIE_Online Cov | MAPIE_Online Wid | MAPIE_Batch Cov | MAPIE_Batch Wid |
|---|---|---|---|---|---|---|---|
| MLP | 876 | 0.896±0.000 | 209.6±0.9 | 0.900±0.000 | 214.5±0.0 | 0.727±0.000 | 101.6±0.0 |
| MLP | 1654 | 0.903±0.000 | 196.4±0.6 | 0.903±0.000 | 197.0±0.0 | 0.778±0.000 | 110.2±0.0 |
| MLP | 2433 | 0.906±0.000 | 196.0±0.4 | 0.907±0.000 | 197.4±0.0 | 0.842±0.000 | 132.0±0.0 |
| RF | 876 | 0.896±0.001 | 257.8±1.6 | 0.904±0.000 | 246.0±0.0 | 0.772±0.000 | 98.0±0.0 |
| RF | 1654 | 0.903±0.001 | 253.7±2.7 | 0.904±0.000 | 251.9±0.0 | 0.824±0.000 | 127.0±0.0 |
| RF | 2433 | 0.912±0.001 | 240.1±1.6 | 0.908±0.000 | 241.7±0.0 | 0.853±0.000 | 162.6±0.0 |
| Ridge | 876 | 0.895±0.001 | 181.0±1.0 | 0.896±0.000 | 171.7±0.0 | 0.736±0.000 | 72.8±0.0 |
| Ridge | 1654 | 0.900±0.000 | 194.4±1.0 | 0.900±0.000 | 175.7±0.0 | 0.825±0.000 | 103.5±0.0 |
| Ridge | 2433 | 0.906±0.000 | 191.5±0.8 | 0.906±0.000 | 181.6±0.0 | 0.859±0.000 | 126.7±0.0 |

### Plots
- `figure2_replication_multivariate.png` — Coverage and width vs training size (multivariate)
- `figure2_replication_univariate.png` — Coverage and width vs training size (univariate)

## 6. Key Findings

### Finding 1: Original GitHub faithfully implements Algorithm 1
The original code matches the paper's mathematical description on all six dimensions inspected.
Our standalone re-implementation produces numerically identical results, confirming this.

### Finding 2: MAPIE has three meaningful algorithmic differences
MAPIE's `method='enbpi'` is EnbPI-inspired but not a faithful reproduction:
- **Block bootstrap** (not i.i.d.) changes OOB membership patterns
- **Mean aggregation** (not quantile) changes the prediction center from conservative to symmetric
- **Signed residuals** (not absolute) enables asymmetric intervals

### Finding 3: Online score updating is essential
MAPIE Batch (no score updates) severely under-covers across all settings.
MAPIE Online (with score updates) achieves coverage comparable to the original.
This confirms EnbPI's core insight: adaptivity through residual updating is what provides valid coverage.

### Finding 4: Despite algorithmic differences, MAPIE Online achieves comparable coverage
The three MAPIE differences partially cancel out. Block bootstrap preserves temporal structure
(arguably beneficial for time series). The net effect on coverage is modest when online updates are used.

### Finding 5: MLP regressor behaves similarly to Ridge and RF
The sklearn MLPRegressor (proxy for the paper's Keras NeuralNet) shows similar coverage-width
tradeoffs, confirming that EnbPI's conformal guarantee is model-agnostic.

## 7. Implications for Benchmarking

When benchmarking against EnbPI in research:
- **Use the original GitHub code** for faithful Algorithm 1 reproduction
- **MAPIE is acceptable** for practical use if online score updates are enabled, but note the algorithmic differences
- **MAPIE Batch mode should never be used** as a proxy for EnbPI — it lacks the essential online update mechanism

## 8. Limitations

- Sections 5.3 (conditional coverage with missing data) and 5.4 (anomaly detection) were not replicated
  as they are less relevant to the benchmarking context
- RNN regressor was omitted (requires Keras/TensorFlow); MLP serves as a neural network proxy
- Only the Solar Atlanta dataset was used (primary dataset in the paper's Section 5.1-5.2)

## 9. References

1. Xu, C. & Xie, Y. (2021). "Conformal Prediction Interval for Dynamic Time-Series." ICML.
2. Original GitHub: https://github.com/hamrel-cxu/EnbPI
3. MAPIE documentation: https://mapie.readthedocs.io/
