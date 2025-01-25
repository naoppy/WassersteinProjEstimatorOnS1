import numpy as np
import numpy.typing as npt
from scipy.stats import vonmises


def cumsum_hist(mu: float, kappa: float, bin_num) -> npt.NDArray[np.float64]:
    """[-pi, pi] の間を bin_num (=D) 等分した区間でのcdfの値を返す
    [F(i/D)] i=0,1,...,D
    """
    dist = vonmises(loc=mu, kappa=kappa)
    x = np.linspace(-np.pi, np.pi, bin_num + 1)
    y = dist.cdf(x) - dist.cdf(-np.pi)

    assert len(y) == bin_num + 1
    assert abs(y[0] - 0.0) < 1e-7
    assert abs(y[-1] - 1.0) < 1e-7
    return y


def cumsum_hist_data(data: npt.NDArray[np.float64], bin_num) -> npt.NDArray[np.float64]:
    """[-pi, pi] の間を bin_num (=D) 等分した区間でのデータのcdfの値を返す
    [F(i/D)] i=0,1,...,D
    """
    n = len(data)
    data_hist = np.zeros(bin_num + 1)
    for x in data:
        data_hist[
            np.clip(int((x + np.pi) / (2 * np.pi) * bin_num) + 1, 1, bin_num)
        ] += 1
    data_cumsum_hist = np.cumsum(data_hist) / n

    assert abs(data_cumsum_hist[0] - 0.0) < 1e-7
    assert abs(data_cumsum_hist[-1] - 1.0) < 1e-7
    return data_cumsum_hist
