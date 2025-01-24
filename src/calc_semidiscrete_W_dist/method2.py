import numpy as np
import numpy.typing as npt


def calc_semidiscreate_W_dist(
        x_hist_cumsum: npt.NDArray[np.float],
        y_hist_cumsum: npt.NDArray[np.float],
    ) -> float:
    """半離散ケースでの1-Wasserstein距離を計算する
    計算量は O(n) である。

    Args:
        x_hist_cumsum (npt.NDArray[np.float]): F(i/N) (i=0, 1, ..., N-1)
        y_hist_cumsum (npt.NDArray[np.float]): F(i/N) (i=0, 1, ..., N-1)

    Returns:
        float: 1-Wasserstein距離
    """
    assert len(x_hist_cumsum) == len(y_hist_cumsum)
    D = len(x_hist_cumsum)
    levmed = np.median(x_hist_cumsum - y_hist_cumsum)
    return np.sum(np.abs(x_hist_cumsum - y_hist_cumsum - levmed)) / D
