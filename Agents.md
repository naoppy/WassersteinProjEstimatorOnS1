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
│   └── [subfolders]/      # CSV tables organized by experiment type
├── docs/                  # LaTeX math proofs and benchmarks
│   ├── ai_guidelines.md   # Historic guidelines
│   ├── quantile_sampling_comparison.md  # Detailed hybrid method analysis
│   └── calc_dists/        # Analytical calculations documentation (TeX/PDF)
├── src/                   # Source code
│   ├── distributions/     # von Mises, Wrapped Cauchy, Sine-Skewed von Mises
│   ├── method/            # Wasserstein distance calculation routines
│   ├── experiments/       # Simulation execution scripts (ex6 to ex11)
│   ├── plots/             # LaTeX TikZ compilation scripts, and visualizers
│   └── utils/             # Grid generation, quantile sampling, and distance helpers
└── verification/          # Scripts verifying math, algorithms, and loss landscapes
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
* **General LaTeX Compilation**: For standard document builds, compile `.tex` files containing equations/math directly using standard compiler commands:
  ```bash
  lualatex docs/calc_dists/calc_dists.tex
  ```
* **Plotting CSV data via TikZ**: When compiling `.tex` files that plot experimental CSV data located under `./data/` into PDF and SVG formats, use the custom `csv2tikz` wrapper module:
  ```bash
  uv run python -m src.plots.csv2tikz data/vonmises_MSE/ex6.tex
  ```
  *(Note: The script runs `platex`, `dvipdfmx`, and `pdftocairo` under the hood to resolve relative table paths and output SVG files).*

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

AI agents must ensure correctness, stability, and speed when implementing mathematical equations:

### 5.1 Numerical Stability under Sharp Distributions
* When distributions become extremely concentrated/sharp (e.g., concentration parameters like $\kappa$ or $\rho$ are very large), direct calculations can lead to numerical overflow, underflow, or division by zero.
* **Guideline**: Implement guardrails such as using exponentially-scaled functions (e.g., `scipy.special.ive` for Bessel functions), clipping denominators, or using alternative analytical approximations under extreme parameter values.

### 5.2 Performance Optimization (Numba)
* Optimization loops and grid searches are computationally intensive. For critical computation paths (such as grid searches, CDF evaluations, or custom integration routines), propose and implement high-performance code using Numba's JIT compilation (`@numba.njit`) to avoid Python runtime overhead.

### 5.3 Coding Guidelines & Mathematical Comments
* Mathematical code must be highly maintainable. When implementing complex numerical algorithms or mathematical formulas (such as series expansions, KL divergences, or Wasserstein distances), write descriptive, clear comments explaining the equations, boundary checks, and mathematical rationale.

---

## 6. Research & Data Integrity Rules

* **Reproducibility**: Always initialize a random seed at the top of experiment scripts (e.g., `np.random.seed(42)`) to ensure simulation results are consistent.
* **Data Preservation**: Under no circumstances should existing CSV files or generated TikZ/SVG/PDF figures under `data/` be deleted or modified, as these contain historical simulation experiment runs. If you need to re-run simulations, output to a temporary or newly named file unless explicitly directed to overwrite.
* **Hardware profile**: The codebase is optimized for multi-core CPUs via `joblib.Parallel` and `scipy`. Avoid introducing GPU dependencies.
