# AI Guidelines & Repository Documentation

Welcome! This document is designed for AI coding assistants working in this repository. It provides details on the codebase's objectives, its architecture, execution instructions, and standards for conducting experiments.

---

## 1. Project Overview & Background
- **Research Topic**: Wasserstein projection estimators for circular distributions (B4 graduation thesis experiments).
- **Target Paper**: [Wasserstein projection estimators for circular distributions](https://arxiv.org/abs/2510.18367) (arXiv:2510.18367).
- **Goal**: Implement and evaluate projection estimators on circular probability distributions (e.g. von Mises, wrapped Cauchy) using the Wasserstein metric.
- **Hardware Profile**: CPU-only computation (uses multi-core CPU parallelization via `parfor` and `scipy`). No GPU required. Computation times can be high for large sample/grid sizes.

---

## 2. Directory Structure and Modules

### `src/` (Source Code)

#### `src/distributions/`
Contains Python implementations of circular probability distributions:
- `vonmises.py`: von Mises distribution probability density function (PDF), cumulative distribution function (CDF), sampling, and fitting.
- `wrappedcauchy.py`: Wrapped Cauchy distribution PDF, CDF, sampling, and fitting.
- `cauchy.py`: Standard/Wrapped Cauchy distributions utilities.
- `sine_skewed_vonmises.py`: Sine-skewed von Mises distribution.

#### `src/calc_semidiscrete_W_dist/`
Implements the core algorithms to calculate semi-discrete Wasserstein distance on a circle:
- `method1.py`: First algorithm variant.
- `method2.py`: Second algorithm variant.

#### `src/experiments/`
Entry-point scripts for conducting simulations. All experiment scripts are numbered (e.g., `ex1` to `ex11`) and should be executed using Python's module format:
- `ex1_vonmises_method1.py` & `ex2_vonmises_method2.py`: Validation scripts for Von Mises.
- `ex3_wrapped_cauchy.py`: Validation script for Wrapped Cauchy.
- `ex4_vonmises_plot.py` & `ex5_wrapcauchy_plot.py`: Plotting distribution shapes.
- `ex6_vonmises_MSE.py` & `ex65_vonmises_MSE2.py`: Mean Squared Error (MSE) simulations for Von Mises.
- `ex7_wrapcauchy_MSE.py` & `ex75_wrapcauchy_MSE2.py`: MSE simulations for Wrapped Cauchy.
- `ex8_vonmises_mix_MSE.py` & `ex85_vonmises_mix_MSE2.py`: MSE simulations for mixtures of Von Mises.
- `ex9_ss_vonmises_MSE.py` & `ex95_ss_vonmises_MSE2.py`: MSE simulations for sine-skewed Von Mises.
- `ex10_model_misspecification.py` & `ex10_model_misspecification2.py`: Misspecified model simulations.
- `ex11_time.py`: Comparison of execution/computation times across methods.

#### `src/plots/` & `src/misc/`
- Plotting utilities, converters (`txt2csv.py`, `csv_visualizer.py`), and validation test scripts (e.g., `vonmises_cdf_test.py`).

### `data/` (Experiment Artifacts)
- **All simulation results** (CSV datasets, text reports) and **generated visualization figures** are saved in this directory.
- `data/csv_data/`: Subdirectory organizing CSV data by experiment type (e.g., `vonmises_MSE`, `wrapcauchy_MSE`, `time_comparison.csv`).

---

## 3. Environment & Execution Guidelines

### Package Management
- The project runs in a Python >= 3.14 environment managed by `uv`.
- Virtual environment directory: `.venv/`
- Virtual environment python executable: `.venv/Scripts/python.exe` (Windows)

To install or sync dependencies:
```bash
uv sync
```

### Running Scripts
Always run python scripts in module mode using `uv run python -m src.experiments.<script_name_without_py>`. Do **NOT** run them directly, and do **NOT** append `.py`.
- **Correct**:
  ```bash
  uv run python -m src.experiments.ex6_vonmises_MSE -O
  ```
- **Incorrect**:
  ```bash
  uv run python src/experiments/ex6_vonmises_MSE.py
  ```
  *(This will result in relative package path errors like `ImportError` or `ModuleNotFoundError`)*.

- Use the `-O` optimization flag for long runs as it enables standard optimizations (like ignoring asserts) which might speed up heavy math loops.

---

## 4. Development Workflow & Best Practices

1. **Ruff Formatting**:
   - Ruff is configured as the workspace formatter.
   - When modifying or creating Python scripts, you must format and lint **only the changed files**. Avoid running a global linter check across the entire repo, as legacy code might trigger style warnings.
   - To format a specific file:
     ```bash
     uv run ruff format src/experiments/my_experiment.py
     ```
   - To lint check a specific file:
     ```bash
     uv run ruff check src/experiments/my_experiment.py
     ```

2. **Data Preservation**:
   - Ensure you do not delete or modify existing CSVs in the `data/` folder unless requested by the user, as these are results of B4 graduation thesis experiments.
   - Save any new outputs into `data/` using descriptive, numbered filenames corresponding to the experiments.

3. **Reproducibility**:
   - Always set appropriate random seeds (e.g., via `numpy.random.seed`) at the top of experiment modules to ensure research results are fully reproducible.
