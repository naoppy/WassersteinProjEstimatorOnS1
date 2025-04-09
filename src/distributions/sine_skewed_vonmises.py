import numpy as np
import numpy.typing as npt
from scipy.stats import vonmises
from scipy.special import i0, i1, iv


def pdf(x: npt.NDArray[np.float64], mu: float, lam: float) -> npt.NDArray[np.float64]:
    """Sine-skewed von Mises分布の確率密度関数を計算する


    Args:
        x: 確率密度関数を計算する点
        mu: 平均
        lam: 摂動項のパラメータ

    Returns:
        確率密度関数の値
    """
