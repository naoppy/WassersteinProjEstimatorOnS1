import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wrapcauchy


def main():
    mu = np.pi
    rho = 0.7
    dist = wrapcauchy(loc=mu, c=rho)
    x = np.linspace(0, 2*np.pi, 1000, endpoint=True)
    y = dist.cdf(x)
    plt.plot(x, y)
    plt.show()
    y = dist.pdf(x)
    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    main()
