import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def main():
    mu = -np.pi / 2
    kappa = 0.6
    vonmises = stats.vonmises(loc=mu, kappa=kappa)
    x = np.linspace(-np.pi, np.pi, 1001)
    # pdfの表示
    y = vonmises.pdf(x)
    plt.plot(x, y)
    plt.show()

    # ヒストグラム近似の図
    plt.subplot(projection="polar")
    plt.plot(x, y)
    bin_num = 10
    x2 = np.linspace(-np.pi, np.pi, bin_num + 1)
    y2 = [vonmises.cdf(i) - vonmises.cdf(i - (2 * np.pi / bin_num)) for i in x2]
    plt.bar(x2, y2, width=0.1, color="red")
    print(np.sum(y2))
    print(y2)
    plt.show()

    # mu1 = 0.3
    # kappa1 = 5
    # x = np.linspace(-np.pi, np.pi, 10001)
    # y = (
    #     stats.vonmises(loc=mu1, kappa=kappa1).pdf(x) / 2
    #     + stats.vonmises(loc=mu1 + np.pi * 3 / 4, kappa=kappa1).pdf(x) / 2
    # )
    # fig = plt.figure(figsize=(12, 6))
    # left = plt.subplot(121)
    # right = plt.subplot(122, projection="polar")
    # left.plot(x, y)
    # left.grid(True)
    # left.set_title("cartesian plot")
    # right.plot(x, y)
    # right.set_title("polar plot")
    # plt.show()


if __name__ == "__main__":
    main()
