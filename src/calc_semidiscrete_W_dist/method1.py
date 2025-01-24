import numpy.typing as npt
import ot


def method1(
        x_empirical: npt.NDArray[float], # type: ignore
        y_sampling: npt.NDArray[float], # type: ignore
        p: int
    ) -> float:
    """半離散ケースでのp-Wasserstein距離を計算する
    内部でソートをするので O(n log n) かかる。

    Args:
        x_empirical (npt.NDArray[np.float]): [0, 1)のデータ
        y_sampling (npt.NDArray[np.float]): [0, 1)の真の分布からのサンプリングによって得られたデータ
        p (int): p-Wasserstein距離のp

    Returns:
        float: p-Wasserstein距離
    """
    return ot.binary_search_circle(x_empirical, y_sampling, p=p, log=False)

