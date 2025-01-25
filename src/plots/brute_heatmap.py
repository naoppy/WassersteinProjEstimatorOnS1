from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt


def plot_heatmap(ret, param_names: Tuple[str, str]):
    x_0, fval, grid, Jout = ret
    print(f"opt x: {x_0}, fval: {fval}")
    dim = np.shape(grid)[0]
    assert dim == 2  # パラメータが2つの場合のみ対応
    fig, ax = plt.subplots(1)
    image = ax.pcolormesh(grid[0], grid[1], Jout)
    fig.colorbar(image)
    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    plt.show()
