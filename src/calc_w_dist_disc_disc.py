import matplotlib.pylab as plt
import numpy as np
import ot
from scipy.special import iv


def pdf_von_Mises(theta, mu, kappa):
    pdf = np.exp(kappa * np.cos(theta - mu)) / (2.0 * np.pi * iv(0, kappa))
    return pdf


def print_result(cost, log_dict, p: int):
    print(f"p: {p}, cost: {cost[0]}, theta: {log_dict['optimal_theta'][0]}")
    if len(log_dict) != 1:
        print(log_dict)


def main():
    t = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    n = 100
    mu1 = 1
    kappa1 = 20
    mu2 = 0
    kappa2 = kappa1
    x1 = np.random.vonmises(mu1, kappa1, size=n) + np.pi
    x2 = np.random.vonmises(mu2, kappa2, size=n) + np.pi
    plt.figure()
    plt.plot(np.cos(t), np.sin(t), c="k")
    plt.xlim(-1.25, 1.25)
    plt.ylim(-1.25, 1.25)
    plt.scatter(np.cos(x1), np.sin(x1), c="b")
    plt.scatter(np.cos(x2), np.sin(x2), c="r")

    cost1, log_dict1 = ot.binary_search_circle(x1, x2, p=1, log=True)
    cost2, log_dict2 = ot.binary_search_circle(x1, x2, p=2, log=True)
    cost3, log_dict3 = ot.binary_search_circle(x1, x2, p=8, log=True)
    print(f"n = {n}")
    print_result(cost1, log_dict1, p=1)
    print_result(cost2, log_dict2, p=2)
    print_result(cost3, log_dict3, p=8)

    # print if you want to see the circular distribution
    # plt.show()


if __name__ == "__main__":
    main()
