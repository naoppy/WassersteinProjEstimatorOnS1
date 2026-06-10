# AI Instructions & Repository Rules (Agents.md)

This file contains rules, guidelines, and context for AI coding assistants (like Antigravity, Cline, Cursor, etc.) when working on this B4 graduation thesis repository.

---

## 1. Project Context & Purpose
- **Goal**: Research and implement "Wasserstein projection estimators for circular distributions".
- **Target Paper**: Implementation of [Wasserstein projection estimators for circular distributions](https://arxiv.org/abs/2510.18367).
- **Environment**: Python (>= 3.14) managed exclusively via `uv`.

## 2. Directory Structure & File Roles
- `src/`: Core Python source code directory.
  - `src/experiments/`: Execution scripts for running experiments (e.g. von Mises distribution MSE, wrapped Cauchy distribution, time comparisons). **All experiment entry points are here.**
  - `src/calc_semidiscrete_W_dist/`: Functions and modules for calculating semi-discrete Wasserstein distances.
  - `src/distributions/`: Implementations of circular distributions (e.g., wrapped Cauchy, von Mises).
  - `src/misc/` & `src/plots/`: Utilities and plotting code.
- `data/`: CSV data and generated figures.
  - `data/csv_data/`: Contains intermediate outputs and final experiment results in CSV format.
  - Plots and images should be saved here or in designated folders requested by the user.
- `pyproject.toml` & `uv.lock`: Dependency definitions and configuration. Ruff is configured here.

## 3. Execution Commands
- **Dependency & Sync**: Always use `uv sync` to set up or update dependencies.
- **Running Experiment Scripts**:
  - Always run scripts as modules using `uv run python -m src.experiments.<module_name>` (do NOT include `.py` suffix).
  - Example: `uv run python -m src.experiments.ex6_vonmises_MSE -O`
  - Running scripts directly using raw `python src/experiments/ex6_vonmises_MSE.py` will cause `ModuleNotFoundError` for relative/absolute sibling imports.
  - Utilize the `-O` (optimize) flag where applicable, as computation times can be long.

## 4. Coding Style & Formatting Rules
- **Linter & Formatter**: Ruff is configured in `pyproject.toml`.
- **Ruff execution**:
  - **Crucial**: Only run formatting/checks on modified or newly created files. Avoid running broad formatting/checks on the entire repository unless requested, to prevent noise from existing codebase styling.
  - Format modified files: `uv run ruff format <modified_file_paths>`
  - Lint modified files: `uv run ruff check <modified_file_paths>`
- **Styling Details**:
  - Quotes: Use double quotes `"` for strings.
  - Line length limit: 88 characters.
  - Indent: 4 spaces.
