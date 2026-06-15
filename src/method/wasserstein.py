from typing import Optional

import numpy as np
import numpy.typing as npt
import ot


def circular_wasserstein_from_samples(
    x_empirical: npt.NDArray[float],  # type: ignore
    y_sampling: npt.NDArray[float],  # type: ignore
    p: int,
    sorted: Optional[bool] = False,
) -> float:
    """サンプルデータから円周上のp-Wasserstein距離を計算する。

    内部でソートをするので O(n log n) かかる。
    sortedがTrueの場合は O(n)。

    Args:
        x_empirical (npt.NDArray[float]): [0, 1)のデータ
        y_sampling (npt.NDArray[float]): [0, 1)の真の分布からのサンプリング配列
        p (int): p-Wasserstein距離のp
        sorted (Optional[bool]): ソート済みかどうかのフラグ

    Returns:
        float: p-Wasserstein距離
    """
    return ot.binary_search_circle(
        x_empirical, y_sampling, p=p, log=False, require_sort=not sorted
    )


def circular_w1_from_cumsums(
    x_hist_cumsum: npt.NDArray[float],  # type: ignore
    y_hist_cumsum: npt.NDArray[float],  # type: ignore
) -> float:
    """等分割ヒストグラムの累積和から円周上の1-Wasserstein距離を計算する。

    計算量は O(n) である。

    Args:
        x_hist_cumsum (npt.NDArray[float]): F(i/N) (i=0, 1, ..., N-1)
        y_hist_cumsum (npt.NDArray[float]): G(i/N) (i=0, 1, ..., N-1)

    Returns:
        float: 1-Wasserstein距離
    """
    assert len(x_hist_cumsum) == len(y_hist_cumsum)
    D = len(x_hist_cumsum)
    levmed = np.median(x_hist_cumsum - y_hist_cumsum)
    return np.sum(np.abs(x_hist_cumsum - y_hist_cumsum - levmed)) / D
