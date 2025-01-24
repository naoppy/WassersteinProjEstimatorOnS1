import numpy as np
import numpy.typing as npt
from scipy.stats import vonmises


def cumsum_hist(mu: float, kappa: float, bin_num) -> npt.NDArray[np.float64]:
    """bin_num等分した区間でのcdfの値を返す
    [F(i/D)] i=0,1,...,D
    """
    dist = vonmises(loc=mu, kappa=kappa)
    x = np.linspace(-np.pi, np.pi, bin_num + 1)
    y = dist.cdf(x) - dist.cdf(-np.pi)

    assert len(y) == bin_num + 1
    assert abs(y[0] - 0.0) < 1e-7
    assert abs(y[-1] - 1.0) < 1e-7
    return y

