""""
対応するファイル: ./data/実験7_再実験.txt
"""

import matplotlib.pyplot as plt
import numpy as np

from ..distributions import wrapedcauchy


def main():
    true_mu = np.pi / 8
    true_rho = 0.7
    N = [100, 500, 1000, 5000, 10000]
    MLE_okamura_mu_mse = [
        0.0048946792849701515,
        0.001040989034400006,
        0.0006487659517055001,
        0.00011879783361490747,
        4.750049421722644e-05,
    ]
    MLE_okamura_rho_mse = [
        0.0024543682618964795,
        0.0004939665906366113,
        0.0002580234075338128,
        4.748406674090364e-05,
        2.4602075358648317e-05,
    ]
    MLE_Kent_mu_mse = [
        0.09605627270551773,
        0.09472573060858208,
        0.0939400524358292,
        0.09406819552646314,
        0.09420881523792778,
    ]
    MLE_Kent_rho_mse = [
        0.10234616948739941,
        0.09285720058048769,
        0.09395051909508309,
        0.09447946736375559,
        0.09424316475035897,
    ]
    method1_mu_mse = [
        0.04034090395000647,
        0.006679415156918051,
        0.0027495499244148723,
        0.00044548471642139286,
        0.0001723104204312281,
    ]
    method1_rho_mse = [
        0.00969109238549894,
        0.0015778230459309011,
        0.0007409083172461962,
        0.00024513185239323303,
        7.701963885382916e-05,
    ]
    method2_mu_mse = [
        0.0038671699741998104,
        0.0005614792510981163,
        0.0003611461293781733,
        6.367869971534874e-05,
        3.1352945388194426e-05,
    ]
    method2_rho_mse = [
        0.0015524113145540224,
        0.00024796371064772515,
        0.00016940247555217064,
        3.404897086664361e-05,
        1.661390857181569e-05,
    ]
    method3_mu_mse = [
        0.0069948569533737105,
        0.0008732879505489276,
        0.0005076431986732359,
        0.0001077102772467967,
        4.7444885368421625e-05,
    ]
    method3_rho_mse = [
        0.0024353126669093604,
        0.00039525320220662794,
        0.0002574877876091715,
        5.5241671907054404e-05,
        2.634635709054017e-05,
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
    plt.plot(np.log10(N), np.log10(MLE_Kent_mu_mse), label="MLE by Kent", marker="o")
    plt.plot(
        np.log10(N),
        np.log10(method1_mu_mse),
        label="W2-estimator by method1",
        marker="o",
    )
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
    plt.plot(np.log10(N), np.log10(MLE_Kent_rho_mse), label="MLE by Kent", marker="o")
    plt.plot(
        np.log10(N),
        np.log10(method1_rho_mse),
        label="W2-estimator by method1",
        marker="o",
    )
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
