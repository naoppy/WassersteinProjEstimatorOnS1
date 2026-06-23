# Wasserstein Projection Estimators for Circular Distributions

This repository contains the official Python implementation of the experiments for the research paper:  
**[Wasserstein projection estimators for circular distributions](https://arxiv.org/abs/2510.18367)** (arXiv:2510.18367).

The codebase implements, evaluates, and visualizes Wasserstein projection estimators on circular probability distributions (e.g., von Mises, Wrapped Cauchy, Sine-Skewed von Mises), and compares their performance and robustness against Maximum Likelihood Estimation (MLE) and other estimators.

---

## Table of Contents
1. [Overview (Repository Structure)](#overview-repository-structure)
2. [Environment & Setup](#environment--setup)
3. [How to Run Experiments](#how-to-run-experiments)
4. [Summary of Experiments](#summary-of-experiments)
5. [Calculation Methods](#calculation-methods)
6. [License](#license)

---

## Overview (Repository Structure)

```
.
├── pyproject.toml         # Python dependencies and Ruff configuration
├── uv.lock                # Locked dependencies (handled by uv)
├── README.md              # Project documentation (this file)
├── Agents.md              # Guidelines and context for AI agents
├── data/                  # Experiment output datasets and figures
│   └── [subfolders]/      # CSV results organized by experiment type
├── docs/                  # LaTeX explanations and benchmark records
│   ├── ai_guidelines.md
│   ├── quantile_sampling_comparison.md
│   └── calc_dists/        # PDF & LaTeX source for analytical math
├── src/                   # Python source code
│   ├── distributions/     # Circular distributions (von Mises, Wrapped Cauchy, Sine-Skewed)
│   ├── method/            # Wasserstein distance calculation functions
│   ├── experiments/       # Entry-point scripts for all simulation experiments (ex6-ex11)
│   ├── plots/             # CSV converters, LaTeX TikZ compilers, and matplotlib plotters
│   └── utils/             # Quantile sampling, CDF histogram generation, and distance helpers
└── verification/          # Scripts to verify math correctness, estimation algorithms, and loss landscapes
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

> [!NOTE]
> Basic algorithm validation and loss landscape visualization scripts (formerly `ex1`-`ex5`) have been refactored and moved to the [verification/](./verification/) directory to keep the core experiment suite clean and focused on publication results.

| Experiment | Filename | Distribution | Independent Variable | Dependent Variable / Comparison Target |
| :--- | :--- | :--- | :--- | :--- |
| **ex6** | [ex6_vonmises_MSE.py](./src/experiments/ex6_vonmises_MSE.py) | von Mises | Sample Size $N \in [10^2, 10^5]$ | MSE of MLE, W1 (equal div), W1 (quantile), and W2 (quantile) |
| **ex65** | [ex65_vonmises_MSE2.py](./src/experiments/ex65_vonmises_MSE2.py) | von Mises | Concentration $\kappa \in [0.5, 500]$ | MSE of MLE, W1 (equal div), and W2 (quantile) at $N=10^5$ |
| **ex7** | [ex7_wrapcauchy_MSE.py](./src/experiments/ex7_wrapcauchy_MSE.py) | Wrapped Cauchy | Sample Size $N \in [10^2, 10^5]$ | MSE of MLE (Kent), W1 (equal div), and W2 (quantile) |
| **ex75** | [ex75_wrapcauchy_MSE2.py](./src/experiments/ex75_wrapcauchy_MSE2.py) | Wrapped Cauchy | Concentration $\rho \in [0.1, 0.9]$ | MSE of MLE (Kent), W1 (equal div), and W2 (quantile) at $N=10^5$ |
| **ex8** | [ex8_vonmises_mix_MSE.py](./src/experiments/ex8_vonmises_mix_MSE.py) | Mixture (vM + Uniform) | Sample Size $N \in [10^2, 10^5]$ | Robustness MSE of MLE, W1 (equal div), W2 (quantile), and density-based estimators |
| **ex85** | [ex85_vonmises_mix_MSE2.py](./src/experiments/ex85_vonmises_mix_MSE2.py) | Mixture (vM + Uniform) | Noise Rate $\gamma \in [0.0, 0.5]$ | Robustness MSE of MLE, W1 (equal div), and W2 (quantile) at $N=10^4$ |
| **ex9** | [ex9_ss_vonmises_MSE.py](./src/experiments/ex9_ss_vonmises_MSE.py) | Sine-Skewed von Mises | Sample Size $N \in [10^2, 10^5]$ | MSE of MLE and W1 (equal div) |
| **ex95** | [ex95_ss_vonmises_MSE2.py](./src/experiments/ex95_ss_vonmises_MSE2.py) | Sine-Skewed von Mises | Skewness $\lambda \in [-1, 1]$ | MSE of MLE and W1 (equal div) at $N=10^5$ |
| **ex10** | [ex10_model_misspecification.py](./src/experiments/ex10_model_misspecification.py) | Misspecified (vM data $\rightarrow$ WC fit) | Sample Size $N \in [10^2, 10^5]$ | Misspecification robustness: KL divergence, W1, and W2 distances |
| **ex10_2** | [ex10_model_misspecification2.py](./src/experiments/ex10_model_misspecification2.py) | Misspecified (WC data $\rightarrow$ vM fit) | Sample Size $N \in [10^2, 10^5]$ | Misspecification robustness: KL divergence, W1, and W2 distances |
| **ex11** | [ex11_time.py](./src/experiments/ex11_time.py) | vM & WC | Sample Size $N \in [10^2, 10^5]$ | Computation time benchmark of different estimators |

---

## Calculation Methods

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
* **Aligned 1-Wasserstein Distance $W_1(P_{\text{vM}}, Q_{\text{WC}})$**:
  When the mean directions are aligned ($\mu_P = \mu_Q$) and one distribution is strictly more concentrated than the other such that their CDF difference $F_P(\theta) - F_Q(\theta)$ does not change sign on the open interval $(0, \pi)$, the circular $1$-Wasserstein distance simplifies to the absolute difference of their Mean Absolute Deviations (MAD) on $[-\pi, \pi]$:
  $$W_1(P, Q) = | E_P[|\theta|] - E_Q[|\theta|] |$$
  
  > [!IMPORTANT]
  > **Mathematical Constraint / Verification Assumption**:
  > This analytical simplification holds if and only if there are **no CDF crossings** on the open interval $(0, \pi)$, which is satisfied when $(p(0) - q(0))(p(\pi) - q(\pi)) \le 0$. Under this condition, the optimal circular phase shift $\alpha^*$ that minimizes the Wasserstein distance $\int_{-\pi}^\pi |F_P(\theta) - F_Q(\theta) - \alpha| d\theta$ is exactly $0$ (since the median of the odd function $F_P(\theta) - F_Q(\theta)$ on $[-\pi, \pi]$ is $0$).
  
  Under this constraint, the distance is computed analytically using rapidly converging series:
  $$W_1(P, Q) = \frac{4}{\pi} \left| \sum_{k=0}^\infty \frac{\rho^{2k+1} - \frac{I_{2k+1}(\kappa)}{I_0(\kappa)}}{(2k+1)^2} \right|$$

---

## Useful Tools & Utilities

The repository provides several utilities under `src/plots/` to process data, visualize results, and compile plots:

### 1. LaTeX TikZ/PGFPlots Compiler (`csv2tikz.py`)
Used to compile TikZ-based `.tex` plot files (which load experimental CSV datasets located under `./data/`) into PDF and SVG plots.
* **Usage**:
  ```bash
  uv run python -m src.plots.csv2tikz docs/calc_dists/calc_dists.tex
  ```
  *(Pass the path to the `.tex` file that plots the CSV data. It compiles the file using `platex`, `dvipdfmx`, and converts it to SVG using `pdftocairo`).*

### 2. CSV Visualizer (`csv_visualizer.py`)
A helper utility to quickly plot log-log scales of Sample Size $N$ vs Mean Squared Error (MSE), comparing estimated parameters with the Cramer-Rao Lower Bounds.
* **Usage**:
  ```bash
  uv run python -m src.plots.csv_visualizer
  ```

---

## License

This repository is licensed under the MIT License.  
**Author**: Naoki Otani
