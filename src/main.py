import matplotlib.pyplot as plt
import numpy as np
from scipy.special import i0, i1
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
    # ここから二分探索による数値計算で逆関数を求める
    EPS = 1e-6
    left = EPS
    right = 1000  # これ以上大きくするとベッセル関数が発散(オーバーフロー)してしまう！
    while right - left > EPS:
        mid = (left + right) / 2
        now_value = i1(mid) / i0(mid)
        # print(mid, i0(mid), i1(mid), now_value) # for debug
        if np.abs(now_value - target_value) < EPS:
            break
        elif now_value - target_value > 0:
            right = mid
        else:
            left = mid
    kappa_MLE = mid
    return [mu_MLE, kappa_MLE]


def plot_vonmises(sample, loc, kappa, N):
    # copied from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.vonmises.html
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


def main():
    loc = 0.5 * np.pi  # circular mean
    kappa = 1  # concentration
    N = 10000
    sample = vonmises(loc=loc, kappa=kappa).rvs(N)

    # plot_vonmises(sample, loc, kappa, N)

    T_data = T(sample)
    mu_MLE, kappa_MLE = MLE(T_data, N)
    print(mu_MLE, kappa_MLE)


if __name__ == "__main__":
    main()
