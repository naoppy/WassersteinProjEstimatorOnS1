import numpy as np
import numpy.typing as npt
from scipy.stats import wrapcauchy


def wrapcauchy_true_pdf(
    x: npt.NDArray[np.float64], c: float, loc: float = 0.0, scale: float = 1.0
) -> npt.NDArray[np.float64]:
    """[loc, loc+2*pi*scale] で定義されている pdf を R 全体に拡張"""
    norm_x = loc + np.remainder(x - loc, 2 * np.pi * scale)
    return wrapcauchy.pdf(norm_x, c, loc, scale)


def wrapcauchy_periodic_cdf(
    x: npt.NDArray[np.float64], c: float, loc: float = 0.0, scale: float = 1.0
) -> npt.NDArray[np.float64]:
    """[loc, loc+2*pi*scale] で定義されている cdf を R 全体に拡張"""
    norm_x = loc + np.remainder(x - loc, 2 * np.pi * scale)
    return wrapcauchy.cdf(norm_x, c, loc, scale) + np.floor_divide(
        x - loc, 2 * np.pi * scale
    )


def cumsum_hist(mu: float, rho: float, bin_num) -> npt.NDArray[np.float64]:
    """bin_num等分した区間でのcdfの値を返す
    [F(i/D)] i=0,1,...,D
    """
    x = np.linspace(0, 2 * np.pi, bin_num + 1)
    y = wrapcauchy_periodic_cdf(x, rho, mu, 1) - wrapcauchy_periodic_cdf(0, rho, mu, 1)

    assert len(y) == bin_num + 1
    assert abs(y[0] - 0.0) < 1e-7
    assert abs(y[-1] - 1.0) < 1e-7
    return y
