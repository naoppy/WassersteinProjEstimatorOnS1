# AI Instructions & Repository Rules (Agents.md)

This file contains rules, constraints, architectural context, and execution guidelines for AI coding assistants (like Antigravity, Cline, Cursor, etc.) working on this Wasserstein Projection Estimators research repository.

---

## 1. Project Context & Objectives
* **Goal**: Implement, evaluate, and analyze Wasserstein projection estimators for circular probability distributions.
* **Target Paper**: Implementation of [Wasserstein projection estimators for circular distributions](https://arxiv.org/abs/2510.18367) (arXiv:2510.18367).
* **Research Focus**:
  * Minimizing Wasserstein distance between empirical circular distributions and parametric model families (von Mises, Wrapped Cauchy, Sine-Skewed von Mises).
  * Robustness comparison against Maximum Likelihood Estimation (MLE) under model misspecification or noise contamination.

---

## 2. Directory Structure & File Roles

```
.
├── pyproject.toml         # Dependency configurations and Ruff parameters
├── uv.lock                # Dependency lock file
├── README.md              # Public project documentation
├── Agents.md              # AI instruction manual (this file)
├── data/                  # Experiment data and outputs
│   └── csv_data/          # CSV tables organized by experiment type
├── docs/                  # LaTeX math proofs and benchmarks
│   ├── ai_guidelines.md   # Historic guidelines
│   ├── quantile_sampling_comparison.md  # Detailed hybrid method analysis
│   └── calc_dists/        # Analytical calculations documentation (TeX/PDF)
├── src/                   # Source code
│   ├── distributions/     # von Mises, Wrapped Cauchy, Sine-Skewed von Mises
│   ├── method/            # Wasserstein distance calculation routines
│   ├── experiments/       # Simulation execution scripts (ex1 to ex11)
│   ├── plots/             # converters, LaTeX TikZ compilation scripts, and visualizers
│   └── utils/             # Grid generation, quantile sampling, and distance helpers
└── verification/          # Scripts verifying math calculations and algorithm correctness
```

---

## 3. Execution Constraints & Commands

### 3.1 Python Environment
* Managed exclusively via `uv` with Python version >= 3.14.
* Always run `uv sync` to sync packages/dependencies after updating `pyproject.toml`.

### 3.2 Running Experiments
* **Rule**: Always run experiment scripts in module mode from the repository root. Do **NOT** run files directly with `python path/to/file.py`. Do **NOT** append `.py` to the module name.
* **Command**:
  ```bash
  uv run python -m src.experiments.ex6_vonmises_MSE -O
  ```
* **Optimize Flag (`-O`)**: Always use `-O` for execution runs. This disables assertions and speeds up mathematical calculation loops significantly.

### 3.3 LaTeX & TikZ Compilation
* Compile `.tex` files containing pgfplots/tikz to PDF and SVG using standard compiler commands (such as `lualatex` to build PDFs, followed by standard utilities like `pdftocairo` to generate SVGs if needed).
* **Command Example**:
  ```bash
  lualatex docs/calc_dists/calc_dists.tex
  ```

---

## 4. Coding Style & Quality Controls

### 4.1 Linter & Formatter (Ruff)
* **Crucial Rule**: Run formatting and lint checks **ONLY** on files you modified or created. Do not run global checks on the entire repository to prevent polluting git diffs with styling updates on legacy code.
* **Format command**:
  ```bash
  uv run ruff format src/experiments/my_script.py
  ```
* **Lint command**:
  ```bash
  uv run ruff check src/experiments/my_script.py
  ```

### 4.2 Styling Details
* **Quotes**: Use double quotes `"` for strings.
* **Line Length Limit**: Maximum 88 characters.
* **Indentation**: 4 spaces.
* **Type Hints**: Standard Python type annotations are preferred where possible.

---

## 5. Critical Mathematical & Numerical Context

AI agents must respect the following numerical implementations to ensure stability and correctness:

### 5.1 Numerical Stability of Bessel Functions
* For von Mises distributions, calculating the Bessel ratio $I_v(\kappa) / I_0(\kappa)$ directly leads to float overflow when concentration $\kappa \ge 600$.
* **Rule**: Always use the exponentially-scaled Bessel function utility `scipy.special.ive` for calculating PDF, CDF, MLE, and Fisher Information when $\kappa \ge 600$, implemented via `_bessel_ratio` and `_bessel_ratio_i0` in `src/distributions/vonmises.py`.

### 5.2 Fast Quantile Sampling (Grid + Newton)
* SciPy's default root-finder (`scipy.stats.vonmises.ppf`) is extremely slow for large sample sizes.
* **Rule**: When performing quantile sampling for von Mises in simulations, use `fast_quantile_sampling` from `src/distributions/vonmises.py`. This uses a coarse grid search ($M = 16384$) followed by a 1-step Newton-Raphson correction to yield $\approx 10^{-13}$ precision with up to **1000x speedup**.

### 5.3 Circular $W_1$ from Histograms ($O(M)$)
* Minimizing Wasserstein distance with sample sorting inside optimization loops is slow ($O(N \log N)$).
* **Rule**: Use the equal-division cumulative sum $O(M)$ Wasserstein calculator `circular_w1_from_cumsums` (defined in `src/method/wasserstein.py`) to execute optimization searches efficiently.

### 5.4 Aligned 1-Wasserstein Distance
* For symmetric circular distributions centered at the same location, circular $W_1$ distance is mathematically equivalent to the absolute difference of their Mean Absolute Deviations (MAD).
* Use `w1_aligned_analytical` in `src/utils/dist_utils.py` for direct evaluation.

---

## 6. Research & Data Integrity Rules

* **Reproducibility**: Always initialize a random seed at the top of experiment scripts (e.g., `np.random.seed(42)`) to ensure simulation results are consistent.
* **Data Preservation**: Under no circumstances should existing CSV files or generated TikZ/SVG/PDF figures under `data/` or `data/csv_data/` be deleted or modified, as these contain historical simulation experiment runs. If you need to re-run simulations, output to a temporary or newly named file unless explicitly directed to overwrite.
* **Hardware profile**: The codebase is optimized for multi-core CPUs via `joblib.Parallel` and `scipy`. Avoid introducing GPU dependencies.
