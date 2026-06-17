# Quantile Sampling Performance and Accuracy Comparison

This document records the comparison of different circular quantile sampling methods for the von Mises distribution across varying sample sizes ($N = 10^3, 10^4, 10^5$).

## Methods Evaluated
1. **quantile_sampling (SciPy)**: Standard root-finding method using SciPy's individual `ppf` calls.
2. **fast_quantile_sampling (Grid)**: Direct grid search using a fine grid of size $2^{20} = 1,048,576$.
3. **1-step Newton (Coarse Grid)**: Coarse grid ($M = 16,384$) search with a single Newton-Raphson correction step.
4. **2-step Newton (Coarse Grid)**: Coarse grid ($M = 16,384$) search with two Newton-Raphson correction steps.

---

## Benchmark Results

The following table summarizes execution speed (in seconds) and the maximum absolute numerical error compared to SciPy's exact output:

| Method | N = 1,000 | N = 10,000 | N = 100,000 | Max Abs Error (vs SciPy) |
| :--- | :--- | :--- | :--- | :--- |
| **SciPy (Individual root-finding)** | 1.2465s | 10.5934s | 106.0328s | *0.00e+00 (Reference)* |
| **Grid Search ($2^{20}$ points)** | 0.0507s | 0.0401s | 0.0358s | $\approx 3.00 \times 10^{-6}$ |
| **1-step Newton (Grid $16,384$)** | **0.0023s** | **0.0106s** | **0.1088s** | **$\approx 8.77 \times 10^{-13}$** |
| **2-step Newton (Grid $16,384$)** | 0.0041s | 0.0107s | 0.1162s | $\approx 8.75 \times 10^{-13}$ |

---

## Analysis

1. **Precision limits**:
   - The original **Grid Search** is limited by discretization resolution, resulting in a maximum error of $\approx 3 \times 10^{-6}$.
   - **1-step Newton** reduces this error to the **$10^{-13}$ scale** (essentially matching the double precision truncation limit of the Fourier-Bessel series with $150$ terms).
   - Evaluating a second Newton step does not yield further accuracy improvements because the error is already bounded by the Fourier-Bessel truncation limit.
2. **Execution Speeds**:
   - **Small/Medium Scales ($N \le 10,000$)**: The 1-step Newton hybrid method is **4x to 20x faster** than the original grid method because it avoids evaluating a massive $1,048,576$-point grid.
   - **Large Scales ($N = 100,000$)**: The original Grid Search is slightly faster ($\approx 0.036$s vs $\approx 0.109$s) because it performs only a single grid-wise CDF evaluation. However, the 1-step Newton remains extremely fast ($0.1$ seconds) and provides vastly superior precision.
