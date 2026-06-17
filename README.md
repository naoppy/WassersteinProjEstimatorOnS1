# Wasserstein Projection Estimators for Circular Distributions

This repository contains the Python implementation for research on **Wasserstein projection estimators for circular distributions** (specifically for B4 graduation thesis experiments). The codebase implements, evaluates, and visualizes projection estimators on circular probability distributions using Wasserstein metrics, comparing them with Maximum Likelihood Estimation (MLE) and density-based estimators.

It is based on the target paper: [Wasserstein projection estimators for circular distributions](https://arxiv.org/abs/2510.18367) (arXiv:2510.18367).

---

## Table of Contents
1. [Overview of Calculation Methods](#overview-of-calculation-methods)
2. [Repository Structure](#repository-structure)
3. [Environment & Setup](#environment--setup)
4. [How to Run Experiments](#how-to-run-experiments)
5. [Summary of Experiments](#summary-of-experiments)
6. [Useful Tools & Utilities](#useful-tools--utilities)
7. [License](#license)

---

## Overview of Calculation Methods

### 1. Wasserstein Projection Estimator on the Circle
The Wasserstein projection estimator $\hat{\theta}$ minimizes the circular $p$-Wasserstein distance $W_p$ between the empirical distribution $P_N = \frac{1}{N} \sum_{i=1}^N \delta_{x_i}$ of the sample $\{x_i\}_{i=1}^N \subset [0, 2\pi)$ and the parametric model circular distribution $P_\theta$ (e.g., von Mises, Wrapped Cauchy):
$$\hat{\theta} = \arg\min_{\theta} W_p(P_N, P_\theta)$$

### 2. Fast Circular $W_1$ from Cumulative Sums ($O(M)$)
For equal-division histograms with $M$ bins, the circular $1$-Wasserstein distance can be calculated extremely fast in $O(M)$ time using the median alignment method:
$$W_1(P_N, P_\theta) = \frac{1}{M} \sum_{j=1}^M \left| F_N\left(\frac{j}{M}\right) - F_\theta\left(\frac{j}{M}\right) - \text{median}\left( F_N - F_\theta \right) \right|$$
where $F_N$ is the empirical CDF and $F_\theta$ is the model CDF. This avoids expensive search or sorting inside the optimization loop.

### 3. Fast Hybrid Quantile Sampling (Grid + Newton)
Quantile sampling approximates a continuous distribution $P_\theta$ with $D$ uniform discrete points $y_j = F_\theta^{-1}(\frac{j - 1/2}{D})$. Calculating $F_\theta^{-1}$ using naive SciPy root-finding is computationally heavy.
To achieve up to a **1000x speedup** with double-precision accuracy ($\approx 10^{-13}$ absolute error), a hybrid method is implemented:
1. **Coarse Grid Search**: Precomputes $F_\theta$ on a coarse grid of size $M = 16384$ and finds the containing interval.
2. **Linear Interpolation**: Computes an initial estimate $\theta_0$ within the interval.
3. **1-Step Newton-Raphson Correction**:
   $$\theta \leftarrow \theta - \frac{F(\theta) - q}{f(\theta)}$$
   which achieves machine precision in a single step due to the smoothness of circular distribution CDFs.

### 4. Analytical Calculations for Distance Verification
* **KL Divergence $D_{\text{KL}}(P_{\text{vM}} \parallel Q_{\text{WC}})$**: Computed via a fast-converging Fourier series expansion:
  $$D_{\text{KL}}(P \parallel Q) = \kappa \frac{I_1(\kappa)}{I_0(\kappa)} - \log I_0(\kappa) - \log(1 - \rho^2) - 2 \sum_{n=1}^\infty \frac{\rho^n I_n(\kappa)}{n I_0(\kappa)} \cos(n(\mu_P - \mu_Q))$$
* **KL Divergence $D_{\text{KL}}(Q_{\text{WC}} \parallel P_{\text{vM}})$**: Computed in closed-form:
  $$D_{\text{KL}}(Q \parallel P) = \log I_0(\kappa) - \log(1 - \rho^2) - \kappa \rho \cos(\mu_P - \mu_Q)$$
* **Aligned 1-Wasserstein Distance $W_1(P_{\text{vM}}, Q_{\text{WC}})$**: When mean directions are aligned ($\mu_P = \mu_Q$), the distance simplifies to the difference of their Mean Absolute Deviations (MAD) on $[-\pi, \pi]$:
  $$W_1(P, Q) = \frac{4}{\pi} \left| \sum_{k=0}^\infty \frac{\rho^{2k+1} - \frac{I_{2k+1}(\kappa)}{I_0(\kappa)}}{(2k+1)^2} \right|$$

---

## Repository Structure

```
.
├── pyproject.toml         # Python dependencies and Ruff configuration
├── uv.lock                # Locked dependencies (handled by uv)
├── README.md              # Project documentation (this file)
├── Agents.md              # Guidelines and context for AI agents
├── data/                  # Experiment output datasets and figures
│   └── csv_data/          # CSV results organized by experiment type
├── docs/                  # LaTeX explanations and benchmark records
│   ├── ai_guidelines.md
│   ├── quantile_sampling_comparison.md
│   └── calc_dists/        # PDF & LaTeX source for analytical math
├── src/                   # Python source code
│   ├── distributions/     # Circular distributions (von Mises, Wrapped Cauchy, Sine-Skewed)
│   ├── method/            # Wasserstein distance calculation functions
│   ├── experiments/       # Entry-point scripts for all simulation experiments (ex1-ex11)
│   ├── plots/             # CSV converters, LaTeX TikZ compilers, and matplotlib plotters
│   └── utils/             # Quantile sampling, CDF histogram generation, and distance helpers
└── verification/          # Scripts to verify math correctness and algorithm precision
```

---

## Environment & Setup

### Requirements
* Python >= 3.14
* `uv` (Fast Python package installer and manager)

### Installation
Clone this repository and run the following command to automatically create a virtual environment and install all dependencies:
```bash
uv sync
```

---

## How to Run Experiments

All experiment entry points are inside `src/experiments/`. Always run them as Python modules from the repository root directory. **Do not append `.py` to the module name.**

* **Run an experiment (e.g., ex6)**:
  ```bash
  uv run python -m src.experiments.ex6_vonmises_MSE -O
  ```
  *(The `-O` optimize flag is highly recommended to disable assertions and speed up math execution).*

* **Do not run files directly as scripts**:
  Running `python src/experiments/ex6_vonmises_MSE.py` will result in `ModuleNotFoundError` due to absolute package imports.

---

## Summary of Experiments

| Experiment | Filename | Distribution | Independent Variable | Dependent Variable / Comparison Target |
| :--- | :--- | :--- | :--- | :--- |
| **ex1** | [ex1_vonmises_method1.py](file:///c:/Users/onaok/Desktop/git/TodaiB4Thesis/src/experiments/ex1_vonmises_method1.py) | von Mises | N/A (Validation) | Parameter estimation via minimizing sample-based $W_2$ |
| **ex2** | [ex2_vonmises_method2.py](file:///c:/Users/onaok/Desktop/git/TodaiB4Thesis/src/experiments/ex2_vonmises_method2.py) | von Mises | N/A (Validation) | Parameter estimation via minimizing equal-division $W_1$ |
| **ex3** | [ex3_wrapped_cauchy.py](file:///c:/Users/onaok/Desktop/git/TodaiB4Thesis/src/experiments/ex3_wrapped_cauchy.py) | Wrapped Cauchy | N/A (Validation) | Comparing MLE vs equal-division $W_1$ |
| **ex4** | [ex4_vonmises_plot.py](file:///c:/Users/onaok/Desktop/git/TodaiB4Thesis/src/experiments/ex4_vonmises_plot.py) | von Mises | Grid search | Landscape visualization of Wasserstein distance over ($\mu, \kappa$) |
| **ex5** | [ex5_wrapcauchy_plot.py](file:///c:/Users/onaok/Desktop/git/TodaiB4Thesis/src/experiments/ex5_wrapcauchy_plot.py) | Wrapped Cauchy | Grid search | Landscape visualization of Wasserstein distance over ($\mu, \rho$) |
| **ex6** | [ex6_vonmises_MSE.py](file:///c:/Users/onaok/Desktop/git/TodaiB4Thesis/src/experiments/ex6_vonmises_MSE.py) | von Mises | Sample Size $N \in [10^2, 10^5]$ | MSE of MLE, W1 (equal div), W1 (quantile), and W2 (quantile) |
| **ex65** | [ex65_vonmises_MSE2.py](file:///c:/Users/onaok/Desktop/git/TodaiB4Thesis/src/experiments/ex65_vonmises_MSE2.py) | von Mises | Concentration $\kappa \in [0.5, 500]$ | MSE of MLE, W1 (equal div), and W2 (quantile) at $N=10^5$ |
| **ex7** | [ex7_wrapcauchy_MSE.py](file:///c:/Users/onaok/Desktop/git/TodaiB4Thesis/src/experiments/ex7_wrapcauchy_MSE.py) | Wrapped Cauchy | Sample Size $N \in [10^2, 10^5]$ | MSE of MLE (Kent), W1 (equal div), and W2 (quantile) |
| **ex75** | [ex75_wrapcauchy_MSE2.py](file:///c:/Users/onaok/Desktop/git/TodaiB4Thesis/src/experiments/ex75_wrapcauchy_MSE2.py) | Wrapped Cauchy | Concentration $\rho \in [0.1, 0.9]$ | MSE of MLE (Kent), W1 (equal div), and W2 (quantile) at $N=10^5$ |
| **ex8** | [ex8_vonmises_mix_MSE.py](file:///c:/Users/onaok/Desktop/git/TodaiB4Thesis/src/experiments/ex8_vonmises_mix_MSE.py) | Mixture (vM + Uniform) | Sample Size $N \in [10^2, 10^5]$ | Robustness MSE of MLE, W1 (equal div), W2 (quantile), and density-based estimators |
| **ex85** | [ex85_vonmises_mix_MSE2.py](file:///c:/Users/onaok/Desktop/git/TodaiB4Thesis/src/experiments/ex85_vonmises_mix_MSE2.py) | Mixture (vM + Uniform) | Noise Rate $\gamma \in [0.0, 0.5]$ | Robustness MSE of MLE, W1 (equal div), and W2 (quantile) at $N=10^4$ |
| **ex9** | [ex9_ss_vonmises_MSE.py](file:///c:/Users/onaok/Desktop/git/TodaiB4Thesis/src/experiments/ex9_ss_vonmises_MSE.py) | Sine-Skewed von Mises | Sample Size $N \in [10^2, 10^5]$ | MSE of MLE and W1 (equal div) |
| **ex95** | [ex95_ss_vonmises_MSE2.py](file:///c:/Users/onaok/Desktop/git/TodaiB4Thesis/src/experiments/ex95_ss_vonmises_MSE2.py) | Sine-Skewed von Mises | Skewness $\lambda \in [-1, 1]$ | MSE of MLE and W1 (equal div) at $N=10^5$ |
| **ex10** | [ex10_model_misspecification.py](file:///c:/Users/onaok/Desktop/git/TodaiB4Thesis/src/experiments/ex10_model_misspecification.py) | Misspecified (vM data $\rightarrow$ WC fit) | Sample Size $N \in [10^2, 10^5]$ | Misspecification robustness: KL divergence, W1, and W2 distances |
| **ex10_2** | [ex10_model_misspecification2.py](file:///c:/Users/onaok/Desktop/git/TodaiB4Thesis/src/experiments/ex10_model_misspecification2.py) | Misspecified (WC data $\rightarrow$ vM fit) | Sample Size $N \in [10^2, 10^5]$ | Misspecification robustness: KL divergence, W1, and W2 distances |
| **ex11** | [ex11_time.py](file:///c:/Users/onaok/Desktop/git/TodaiB4Thesis/src/experiments/ex11_time.py) | vM & WC | Sample Size $N \in [10^2, 10^5]$ | Computation time benchmark of different estimators |

---

## Useful Tools & Utilities

The repository provides several utilities to process data, visualize results, and compile reports:

### 1. LaTeX TikZ/PGFPlots Compiler (`csv2tikz.py`)
Compiles LaTeX documents with plots generated from experiment data into PDF and high-quality SVG formats.
```bash
uv run python -m src.plots.csv2tikz docs/calc_dists/calc_dists.tex
```
*Requirement*: Standard LaTeX tools (`platex`, `dvipdfmx`, `pdftocairo`) must be installed on the system PATH.

### 2. Log-to-CSV Converter (`txt2csv.py`)
Parses raw output files from experiments, calculates the diagonal elements of the inverse Fisher Information Matrix (the Cramer-Rao Lower Bounds), and outputs a structured CSV.
```python
from src.plots.txt2csv import ToCSV
ToCSV(methods, params, "input.txt", "output.csv", fisher_bounds)
```

### 3. CSV Visualizer (`csv_visualizer.py`)
A helper utility to quickly plot log-log scales of Sample Size $N$ vs Mean Squared Error (MSE), comparing estimated parameters with the Cramer-Rao Lower Bounds.
```bash
uv run python -m src.plots.csv_visualizer
```

### 4. Fast Quantile Sampling
The core function `fast_quantile_sampling` in `src/distributions/vonmises.py` JIT-compiles via Numba, allowing extremely fast quantile calculations for von Mises. This makes large-scale simulation studies feasible on standard CPUs.

---

## License

This repository is licensed under the MIT License.  
**Author**: Naoki Otani
