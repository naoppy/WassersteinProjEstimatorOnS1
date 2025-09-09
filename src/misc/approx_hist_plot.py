import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import vonmises

def _plot_for_slide():
    """
    vonmises分布をプロットする。その後、ヒストグラムをプロットする。
    """
    # vonmises分布のパラメータ
    mu = 0  # 平均
    # kappa = 1.5  # 尖度
    kappa = 0.7

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection="polar")
    x = np.linspace(-np.pi, np.pi, 1000)
    vonmises_pdf = vonmises.pdf(x, loc=mu, kappa=kappa)
    bin_num = 15
    x2 = np.linspace(-np.pi, np.pi, bin_num)
    weights, _ = np.histogram(
        vonmises.rvs(loc=mu, kappa=kappa, size=100000), bins=bin_num, density=True
    )
    ax.plot(x, vonmises_pdf, label="PDF")
    ax.hist(x2, bins=bin_num, weights=weights, density=True, color="tab:orange", label="Histogram", rwidth=0.7)
    ax.set_yticks([0, 0.15, 0.3])
    ax.set_title("Approximate as Histogram")
    ax.legend(bbox_to_anchor=(0.15, 1.06))
    plt.show()


if __name__ == "__main__":
    _plot_for_slide()