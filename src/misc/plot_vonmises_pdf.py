import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def main():
    vonmises = stats.vonmises(loc=0, kappa=0)
    x = np.linspace(-np.pi, np.pi, 101)
    y = vonmises.pdf(x)
    plt.plot(x, y)
    plt.show()

    # plt.subplot(projection="polar")
    # plt.plot(x, y)
    # plt.show()

    z = vonmises.cdf(x)
    if (np.isnan(z)).any():
        print("nan!")
    if np.isnan(vonmises.cdf(-np.pi)):
        print("nan!!")
    plt.plot(x, z)
    plt.show()


if __name__ == "__main__":
    main()
