import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def main():
    vonmises = stats.vonmises(loc=-np.pi / 2, kappa=0.4)
    x = np.linspace(-np.pi, np.pi, 1000)
    y = vonmises.pdf(x)
    plt.plot(x, y)
    plt.show()

    plt.subplot(projection="polar")
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    main()
