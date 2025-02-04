"""
フォンミーゼス分布のフィッシャー情報量を計算する
"""

import numpy as np
import numpy.typing as npt
from scipy.special import i0, i1, iv


def fisher_info_2x2(kappa: float) -> npt.NDArray[np.float64]:
    return np.array(
        [
            [kappa * i1(kappa) / i0(kappa), 0],
            [0, 1 / 2 + iv(2, kappa) / i0(kappa) - (i1(kappa) / i0(kappa)) ** 2],
        ]
    )
