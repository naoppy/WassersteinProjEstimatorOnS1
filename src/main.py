import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import vonmises


def T(x):
    """フォンミーゼス分布の十分統計量を返す

    Args:
        x: フォンミーゼス分布からのサンプル

    Returns:
        List[float, float]: 十分統計量、[cos, sin] の順
    """
    return [np.sum(np.cos(x)), np.sum(np.sin(x))]


def MLE(T_data, N: int):
    """最尤推定を行う

    Args:
        T_data: 十分統計量
        N(int): サンプル数

    Returns:
        Tuple[float, float]: 最尤推定値、[mu_MLE, kappa_MLE] の順
    """
    mu_MLE = np.arctan2(T_data[1], T_data[0])
    target_value = (T_data[0] * np.cos(mu_MLE) + T_data[1] * np.sin(mu_MLE)) / N
    # TDOO: ここから数値計算で逆関数を求める
    kappa_MLE = 0
    return [mu_MLE, kappa_MLE]


def main():
    loc = 0.5 * np.pi  # circular mean
    kappa = 1  # concentration
    N = 1000
    sample = vonmises(loc=loc, kappa=kappa).rvs(N)

    fig = plt.figure(figsize=(12, 6))
    left = plt.subplot(121)
    right = plt.subplot(122, projection="polar")
    x = np.linspace(-np.pi, np.pi, 500)
    vonmises_pdf = vonmises.pdf(x, loc=loc, kappa=kappa)
    ticks = [0, 0.15, 0.3]
    left.plot(x, vonmises_pdf)
    left.set_yticks(ticks)
    number_of_bins = int(np.sqrt(N))
    left.hist(sample, density=True, bins=number_of_bins)
    left.set_title("Cartesian plot")
    left.set_xlim(-np.pi, np.pi)
    left.grid(True)

    right.plot(x, vonmises_pdf, label="PDF")
    right.set_yticks(ticks)
    right.hist(sample, density=True, bins=number_of_bins, label="Histogram")
    right.set_title("Polar plot")
    right.legend(bbox_to_anchor=(0.15, 1.06))

    plt.show()

    T_data = T(sample)


if __name__ == "__main__":
    main()
