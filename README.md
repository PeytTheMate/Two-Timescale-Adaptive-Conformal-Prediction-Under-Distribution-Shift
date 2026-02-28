# Two-Timescale Adaptive Conformal Prediction Under Distribution Shift

**Independent research by [Peyton Kocher](mailto:howdypeyton@tamu.edu) (Texas A&M University, 2026)**

Adaptive Conformal Inference (ACI) guarantees coverage under distribution shift, but pays for it with massively inflated prediction intervals. This project identifies the root cause, proposes a fix, proves it works theoretically, and validates it on real data.

## The Problem

ACI ([Gibbs & Candes, NeurIPS 2021](https://arxiv.org/abs/2106.00170)) adjusts confidence levels online to maintain coverage under distribution shift. But it never updates the underlying regression model. When the data distribution changes, ACI compensates for the stale model by widening intervals until even bad predictions fall inside them. The intervals are valid, but uninformative.

## The Insight

ACI conflates two timescales of adaptation:
- **Fast:** Calibration correction (adjusting confidence level) needs to happen every step
- **Slow:** Model improvement (refitting the regressor) only needs to happen periodically

By decoupling these, the model stays accurate and intervals stay tight.

## The Method

**Two-Timescale (TTS) Conformal Prediction** runs two nested loops:

| Loop | Frequency | Mechanism | Purpose |
|------|-----------|-----------|---------|
| Fast | Every step | ACI alpha adjustment | Immediate calibration after shifts |
| Slow | Every K steps | Model refit on recent data | Drive model bias toward zero |

## The Theory (Theorem 1)

Coverage regret decomposes into independently optimizable terms:

```
R_T  <=  R_model(K, Delta, S)  +  R_cal(gamma)
```

- **R_model**: Model-bias regret, controlled by refit interval K
- **R_cal**: Calibration regret, controlled by ACI learning rate gamma
- **Optimal refit interval**: K* = O(Delta^{-2/3})

Numerically verified: TTS empirical regret (440) < theoretical bound (759), while ACI accumulates 1,757.

## Results

### Synthetic Data — Experiment A

| Method | Avg Interval Width | Coverage Debt | Wilcoxon p |
|--------|:------------------:|:------------:|:----------:|
| Static CP | 9.66 | 88.2 | — |
| ACI | 10.14 | 32.7 | — |
| DtACI | 10.05 | 34.3 | — |
| **TTS (Ours)** | **2.94** | 38.2 | **0.002** |

### Real Data — Two Datasets, Two Model Classes

| Dataset | Model | TTS/ACI Interval Ratio | Coverage | Checks Passed |
|---------|-------|:----------------------:|:--------:|:-------------:|
| Beijing PM2.5 | Ridge | 0.71x | 0.895 | 3/4 |
| Beijing PM2.5 | GBR | 0.81x | 0.885 | 3/4 |
| Jena Climate | Ridge | **0.17x** | 0.860 | **4/4** |
| Jena Climate | GBR | **0.22x** | 0.881 | **4/4** |

All pairwise comparisons use Wilcoxon signed-rank tests with Holm-Bonferroni correction. Jena Climate achieves p = 0.002 for both model classes.

### When TTS Helps — and When It Doesn't

TTS is beneficial when **model misspecification is reducible**: when refitting the base model on post-shift data actually improves predictions. When the model class cannot capture post-shift patterns, TTS degenerates gracefully to ACI — it does not make things worse.

---

## Repository Structure

```
TTS_Conformal/
├── conformal_experiments_v3.py     # Core: 5 synthetic experiments (A–E), all methods
├── run_real_data.py                # Real-data validation (Beijing PM2.5 + Jena Climate)
├── theory_regret_bound.py          # Theorem 1: statement, proof, numerical verification
│
├── figures/
│   ├── synthetic/                  # Experiments A–E output (5 figures)
│   ├── real_data/                  # Per-dataset, per-model figures + comparisons
│   └── theory/                     # 6-panel theorem verification figure
│
├── data/                           # Datasets (see Reproducing Results below)
│   ├── beijing_features.csv
│   ├── beijing_targets.csv
│   └── jena_climate.csv            # Auto-downloaded on first run
│
├── paper/
│   ├── paper_draft.tex             # 4-page NeurIPS workshop extended abstract
│   └── TTS_Conformal_Inference.pdf # Compiled paper
│
├── reproduce.sh                    # One-command full reproduction
├── requirements.txt                # Python dependencies
└── .gitignore
```

---

## Reproducing Results

**Total runtime: ~5 minutes on a modern CPU. No GPU required.**

### Option 1: One Command

```bash
chmod +x reproduce.sh && ./reproduce.sh
```

### Option 2: Step by Step

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Download Beijing PM2.5 data** (Jena Climate auto-downloads):
```bash
pip install ucimlrepo
python -c "
from ucimlrepo import fetch_ucirepo
d = fetch_ucirepo(id=381)
d.data.features.to_csv('data/beijing_features.csv', index=False)
d.data.targets.to_csv('data/beijing_targets.csv', index=False)
print('Saved to data/')
"
```

**Run experiments:**
```bash
python conformal_experiments_v3.py    # Synthetic experiments A–E  (~3 min)
python theory_regret_bound.py         # Theorem verification       (~2 min)
python run_real_data.py               # Real-data validation       (~80 sec)
```

All figures are written to `figures/`.

---

## Verification Checklist

Each real dataset is evaluated against 4 automated checks:

| Check | Criterion | Beijing Ridge | Beijing GBR | Jena Ridge | Jena GBR |
|-------|-----------|:---:|:---:|:---:|:---:|
| Interval ratio < 0.85 | TTS intervals shorter than ACI | 0.71x | 0.81x | 0.17x | 0.22x |
| Coverage >= 0.85 | TTS maintains adequate coverage | 0.895 | 0.885 | 0.860 | 0.881 |
| Debt within 1.5x ACI | Not catastrophically worse | 1.10x | 1.10x | ~1.0x | ~1.0x |
| Wilcoxon p < 0.05 | Statistically significant | p=0.19 | p=0.19 | **p=0.002** | **p=0.002** |
| **Total** | | **3/4** | **3/4** | **4/4** | **4/4** |

---

## Methods Implemented

| Method | Reference | Role |
|--------|-----------|------|
| Static CP | Vovk et al., 2005 | Baseline (no adaptation) |
| ACI | Gibbs & Candes, NeurIPS 2021 | Online calibration, no model updates |
| DtACI | Gibbs & Candes, JMLR 2024 | Strongly adaptive ACI variant |
| Weighted CP | Barber et al., Ann. Stat. 2023 | Importance-weighted exchangeability |
| CUSUM Detection | Page, 1954 | Shift detection on coverage errors |
| **TTS (Ours)** | This work | Two-timescale: fast calibration + slow refit |

### Relationship to CPTC (Sun & Yu, NeurIPS 2025)

TTS and CPTC are **complementary**: CPTC improves the calibration loop via regime-specific score distributions; TTS improves the model loop via periodic refitting. A natural extension combines both.

---

## Experiments

| ID | Name | What It Tests |
|----|------|--------------|
| A | Two-Timescale Adaptation | Core TTS vs. baselines; regret decomposition visualization |
| B | Importance Weighting | Ablation of recency, density ratio, and clipping under covariate drift |
| C | CUSUM Detection | Shift detection speed; 4.4x faster recovery than ACI |
| D | Real-Data Validation | Beijing PM2.5 + Jena Climate with temporal block bootstrap |
| E | Sensitivity Analysis | Joint (K, gamma) sweep; 65% of settings Pareto-competitive with ACI |

---

## Technical Details

- **Statistical testing:** Paired Wilcoxon signed-rank tests with Holm-Bonferroni correction for all pairwise comparisons
- **Real-data bootstrap:** Temporal block bootstrap (stride=1500, block_size=3000) instead of noise-perturbation pseudo-seeds
- **Feature engineering:** Raw features expanded from 7 to 21 via cyclical encoding, lagged targets, rolling statistics
- **Reproducibility:** Fixed random seeds, deterministic execution, single-file scripts with no hidden state

---

## Citation

```bibtex
@misc{kocher2026twotimescale,
  title   = {Two-Timescale Adaptive Conformal Prediction Under Distribution Shift},
  author  = {Kocher, Peyton},
  year    = {2026},
  note    = {Independent research, Texas A\&M University}
}
```

## License

MIT License. Dataset licenses: UCI Beijing PM2.5 (CC BY 4.0), Jena Climate (public domain).
