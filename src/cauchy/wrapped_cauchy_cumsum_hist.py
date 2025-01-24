import numpy as np
import numpy.typing as npt
from scipy.stats import wrapcauchy


def cumsum_hist(mu: float, rho: float, bin_num) -> npt.NDArray[np.float64]:
    """bin_num等分した区間でのcdfの値を返す
    [F(i/D)] i=0,1,...,D
    """
    dist = wrapcauchy(loc=mu, c=rho)
    x = np.linspace(-np.pi, np.pi, bin_num + 1)
    y = dist.cdf(x) - dist.cdf(-np.pi)

    assert len(y) == bin_num + 1
    assert abs(y[0] - 0.0) < 1e-7
    assert abs(y[-1] - 1.0) < 1e-7
    return y

