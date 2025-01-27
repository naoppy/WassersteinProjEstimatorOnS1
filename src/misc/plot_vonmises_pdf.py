import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def main():
    vonmises = stats.vonmises(loc=0, kappa=1)
    x = np.linspace(-np.pi, np.pi, 101)
    y = vonmises.pdf(x)
    plt.plot(x, y)
    plt.show()

    plt.subplot(projection="polar")
    plt.plot(x, y)
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
