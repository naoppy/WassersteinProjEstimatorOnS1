import numba
import numpy as np
import numpy.typing as npt


@numba.njit(cache=True)
def to_2pi_range(angles: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Map circular angles to [0, 2*pi]."""
    return np.remainder(angles, 2 * np.pi)


def circular_quantile_sampling(ppf_func, sample_num: int) -> npt.NDArray[np.float64]:
    """Generic circular quantile sampling.

    Rolls and sorts the returned values to ensure correct topological
    ordering on [0, 2*pi].
    """
    if sample_num <= 0:
        return np.array([], dtype=np.float64)
    x, step = np.linspace(0, 1, sample_num, endpoint=False, retstep=True)
    x = x + step / 2
    y = ppf_func(x)
    y = to_2pi_range(y)

    if sample_num > 1:
        diffs = np.diff(y)
        min_idx = np.argmin(diffs)
        if diffs[min_idx] < 0:
            i = min_idx + 1
        else:
            i = 0
        y = np.roll(y, -i)

    assert np.all((0 <= y) & (y <= 2 * np.pi))
    assert np.all(np.diff(y) >= 0)
    return y


def cumsum_hist_data(sample, bin_num: int) -> npt.NDArray[np.float64]:
    """empirical CDF generator on equal divisions of [0, 2*pi] domain."""
    sample = to_2pi_range(sample)
    n = len(sample)

    indices = (sample / (2.0 * np.pi) * bin_num).astype(np.intp) + 1
    indices = np.clip(indices, 1, bin_num)

    data_hist = np.bincount(indices, minlength=bin_num + 1)
    data_cumsum_hist = np.cumsum(data_hist) / n

    assert abs(data_cumsum_hist[0] - 0.0) < 1e-7
    assert abs(data_cumsum_hist[-1] - 1.0) < 1e-7
    return data_cumsum_hist
