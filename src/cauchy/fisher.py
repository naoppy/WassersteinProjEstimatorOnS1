"""
巻き込みコーシー分布のフィッシャー情報量を計算する
"""

import numpy as np
import numpy.typing as npt


def fisher_info_2x2(rho: float) -> npt.NDArray[np.float64]:
    rho_p2 = rho * rho
    bunbo = (1 - rho_p2) ** 2
    return np.array(
        [
            [2 * rho_p2 / bunbo, 0],
            [0, 2 / bunbo],
        ]
    )
