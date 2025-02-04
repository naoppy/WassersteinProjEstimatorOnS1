import matplotlib.pyplot as plt
import numpy as np

from ..vonmises.fisher import fisher_info_2x2


def main():
    true_mu = -1.57
    true_kappa = 0.4
    N = [100, 500, 1000, 5000, 10000]
    MLE_mu_mse = [
        0.15724939658265175,
        0.02561219694675373,
        0.01278198115007334,
        0.0023871763147692137,
        0.0011877052930501881,
    ]
    MLE_kappa_mse = [
        0.019592605039796113,
        0.004629387343176226,
        0.0018601594254765194,
        0.0004772945991592503,
        0.00021438215474192746,
    ]
    method1_mu_mse = [
        1.031432638002363,
        0.07029424043908568,
        0.04561552752320401,
        0.007072267712104786,
        0.004729615981022525,
    ]
    method1_kappa_mse = [
        0.29136565446817136,
        0.020964310493274376,
        0.0070253402963549845,
        0.0015724917778899856,
        0.0006508774729222129,
    ]
    method2_mu_mse = [
        0.1547141776387137,
        0.026454532394929547,
        0.012593312391196683,
        0.0025223634068570976,
        0.0012086095499850356,
    ]
    method2_kappa_mse = [
        0.019698553256638218,
        0.004702451004879,
        0.0019637915724737146,
        0.0004894923637704701,
        0.00021822706852973943,
    ]
    fisher_mat = fisher_info_2x2(true_kappa)

    plt.plot(
        np.log10(N), np.log10(MLE_mu_mse), label="MLE", marker="o", linestyle="dashdot"
    )
    plt.plot(
        np.log10(N),
        np.log10(method1_mu_mse),
        label="W2-estimator from method1",
        marker="x",
        linestyle="solid",
    )
    plt.plot(
        np.log10(N),
        -np.log10(fisher_mat[0][0]) - np.log10(N),
        label="Cramer-Rao lower bound",
        marker="^",
        linestyle="solid",
    )
    plt.plot(
        np.log10(N),
        np.log10(method2_mu_mse),
        label="W1-estimator from method2",
        marker="s",
        linestyle="dashed",
    )
    plt.xlabel("log10(N)")
    plt.ylabel("log10(MSE) of mu")
    plt.legend()
    plt.title(f"True parameter: mu={true_mu}, kappa={true_kappa}")
    plt.show()

    plt.plot(
        np.log10(N),
        np.log10(MLE_kappa_mse),
        label="MLE",
        marker="o",
        linestyle="dashdot",
    )
    plt.plot(
        np.log10(N),
        -np.log10(fisher_mat[1][1]) - np.log10(N),
        label="Cramer-Rao lower bound",
        marker="^",
        linestyle="solid",
    )
    plt.plot(
        np.log10(N),
        np.log10(method1_kappa_mse),
        label="W2-estimator from method1",
        marker="x",
        linestyle="solid",
    )
    plt.plot(
        np.log10(N),
        np.log10(method2_kappa_mse),
        label="W1-estimator from method2",
        marker="s",
        linestyle="dashed",
    )
    plt.xlabel("log10(N)")
    plt.ylabel("log10(MSE) of kappa")
    plt.legend()
    plt.title(f"True parameter: mu={true_mu}, kappa={true_kappa}")
    plt.show()


if __name__ == "__main__":
    main()
