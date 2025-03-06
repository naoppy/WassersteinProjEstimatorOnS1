""" "
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
        0.0074690686004980755,
        0.0011628402739038326,
        0.00045767341087098206,
        0.00011349468085199019,
        5.804266503843786e-05,
    ]
    MLE_okamura_rho_mse = [
        0.0026599417738485615,
        0.0004541754237887616,
        0.00026999977255391075,
        5.106894779506723e-05,
        3.3708578842208184e-05,
    ]
    MLE_Kent_mu_mse = [
        0.0033700375386698893,
        0.0006335874845111651,
        0.00020532440995560847,
        5.247294794266492e-05,
        2.9740195137781706e-05,
    ]
    MLE_Kent_rho_mse = [
        0.0012330772511333359,
        0.0002569898046638247,
        0.00011891400223138476,
        2.4888682563840785e-05,
        1.2578287343609827e-05,
    ]
    MLE_direct_mu_mse = [
        0.0033700233389544875,
        0.0006335877176886246,
        0.00020532451079867005,
        5.247282534471107e-05,
        2.9740146249193457e-05,
    ]
    MLE_direct_rho_mse = [
        0.0012330766584881594,
        0.00025698936143804524,
        0.00011891413124673141,
        2.4888639489511244e-05,
        1.2578442008747255e-05,
    ]
    W2_method1_mu_mse = [
        0.022930404907570992,
        0.0030077385435943703,
        0.001206334657515229,
        0.0003653286409253631,
        0.00011987657224842576,
    ]
    W2_method1_rho_mse = [
        0.006238318186887758,
        0.0023530697491451957,
        0.0010488041900364252,
        0.00016892288701187864,
        6.858144183774315e-05,
    ]
    W1_method2_mu_mse = [
        0.003830577064042416,
        0.0008587634246133648,
        0.0002597011308400694,
        8.068191417195986e-05,
        3.512339500772621e-05,
    ]
    W1_method2_rho_mse = [
        0.001891784322587114,
        0.0002901845704954968,
        0.0001589049259547214,
        3.198962177135106e-05,
        1.6343568434074803e-05,
    ]
    W2_method3_mu_mse = [
        0.005800553733858615,
        0.0011781650554296573,
        0.00041422148421900446,
        0.00013473740854743398,
        4.852019623581889e-05,
    ]
    W2_method3_rho_mse = [
        0.002659623996066357,
        0.0003712361434444408,
        0.0002455783022249411,
        4.835086017536754e-05,
        2.4522160368303025e-05,
    ]
    W1_method3_mu_mse = [
        0.0037594225648797464,
        0.0008623113695311706,
        0.0002556188452557837,
        8.070643431326727e-05,
        3.524887277115231e-05,
    ]
    W1_method3_rho_mse = [
        0.0019320929703819678,
        0.0002888280976971131,
        0.00015979553492988967,
        3.198603083061298e-05,
        1.6374893630331123e-05,
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
        np.log10(MLE_direct_mu_mse),
        label="MLE by direct optimization",
        marker="o",
    )
    plt.plot(
        np.log10(N),
        np.log10(W2_method1_mu_mse),
        label="W2-estimator by method1",
        marker="o",
    )
    plt.plot(
        np.log10(N),
        np.log10(W1_method2_mu_mse),
        label="W1-estimator by method2",
        marker="o",
    )
    plt.plot(
        np.log10(N),
        np.log10(W2_method3_mu_mse),
        label="W2-estimator by method3",
        marker="o",
    )
    plt.plot(
        np.log10(N),
        np.log10(W1_method3_mu_mse),
        label="W1-estimator by method3",
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
        np.log10(MLE_direct_rho_mse),
        label="MLE by direct optimization",
        marker="o",
    )
    plt.plot(
        np.log10(N),
        np.log10(W2_method1_rho_mse),
        label="W2-estimator by method1",
        marker="o",
    )
    plt.plot(
        np.log10(N),
        np.log10(W1_method2_rho_mse),
        label="W1-estimator by method2",
        marker="o",
    )
    plt.plot(
        np.log10(N),
        np.log10(W2_method3_rho_mse),
        label="W2-estimator by method3",
        marker="o",
    )
    plt.plot(
        np.log10(N),
        np.log10(W1_method3_rho_mse),
        label="W1-estimator by method3",
        marker="o",
    )
    plt.xlabel("log10(N)")
    plt.ylabel("log10(MSE) of rho")
    plt.legend()
    plt.title(f"True parameter: mu={true_mu}, rho={true_rho}")
    plt.show()


if __name__ == "__main__":
    main()
