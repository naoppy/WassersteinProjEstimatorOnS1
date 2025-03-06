""" "
対応するファイル: ./data/実験7_再実験.txt
"""

import matplotlib.pyplot as plt
import numpy as np

from ..distributions import wrapedcauchy


def main():
    true_mu = np.pi / 8
    true_rho = 0.2
    N = [100, 500, 1000, 5000, 10000]
    MLE_okamura_mu_mse = [
        0.165370134478948,
        0.026610778835372252,
        0.011880847150378255,
        0.002670513336323266,
        0.0011174867019507959,
    ]
    MLE_okamura_rho_mse = [
        0.004353094387881037,
        0.0007563589645292883,
        0.0004048552187668564,
        9.290277303124786e-05,
        4.1071924643349775e-05,
    ]
    method2_mu_mse = [
        5.939831551185607,
        0.02498760726408354,
        0.012191496347006014,
        0.00284353903131702,
        0.0010763002509794608,
    ]
    method2_rho_mse = [
        0.004379350048446914,
        0.0007447747689791547,
        0.0004068799130399253,
        9.712519353720205e-05,
        4.2093063768188895e-05,
    ]
    method3_mu_mse = [
        5.191251011773786,
        0.025371239124917674,
        0.011950841527173527,
        0.002923912144816245,
        0.0010972294146970145,
    ]
    method3_rho_mse = [
        0.004214335237184076,
        0.0007180134824889021,
        0.0004136415695389797,
        9.723989740353548e-05,
        4.220047958789874e-05,
    ]
    fisher_mat = wrapedcauchy.fisher_info_2x2(true_rho)
    plt.plot(
        np.log10(N),
        -np.log10(fisher_mat[0][0]) - np.log10(N),
        label="Cramer-Rao lower bound",
        marker="o",
    )
    plt.plot(
        np.log10(N), np.log10(MLE_okamura_mu_mse), label="MLE by Okamura", marker="o"
    )
    # plt.plot(np.log10(N), np.log10(MLE_Kent_mu_mse), label="MLE by Kent", marker="o")
    plt.plot(
        np.log10(N),
        np.log10(method2_mu_mse),
        label="W1-estimator by method2",
        marker="o",
    )
    plt.plot(
        np.log10(N),
        np.log10(method3_mu_mse),
        label="W2-estimator by method3",
        marker="o",
    )
    plt.xlabel("log10(N)")
    plt.ylabel("log10(MSE) of mu")
    plt.legend()
    plt.title(f"True parameter: mu={true_mu}, rho={true_rho}")
    plt.show()

    plt.plot(
        np.log10(N),
        -np.log10(fisher_mat[1][1]) - np.log10(N),
        label="Cramer-Rao lower bound",
        marker="o",
    )
    plt.plot(
        np.log10(N), np.log10(MLE_okamura_rho_mse), label="MLE by Okamura", marker="o"
    )
    # plt.plot(np.log10(N), np.log10(MLE_Kent_rho_mse), label="MLE by Kent", marker="o")
    plt.plot(
        np.log10(N),
        np.log10(method2_rho_mse),
        label="W1-estimator by method2",
        marker="o",
    )
    plt.plot(
        np.log10(N),
        np.log10(method3_rho_mse),
        label="W2-estimator by method3",
        marker="o",
    )

    plt.xlabel("log10(N)")
    plt.ylabel("log10(MSE) of rho")
    plt.legend()
    plt.title(f"True parameter: mu={true_mu}, rho={true_rho}")
    plt.show()


if __name__ == "__main__":
    main()
