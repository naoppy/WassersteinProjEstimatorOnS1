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
        0.1321428270403234,
        0.02641014684524266,
        0.01220748113996755,
        0.0017611987076882527,
        0.0011826668495757992,
    ]
    MLE_okamura_rho_mse = [
        0.004628303796095025,
        0.0007223482494955518,
        0.0005747933068780715,
        9.49602370892797e-05,
        4.6492846893852045e-05,
    ]
    MLE_Kent_mu_mse = [
        0.1442348928620693,
        0.02652844815995064,
        0.011730158420937513,
        0.001743181562518925,
        0.0011716640732047597,
    ]
    MLE_Kent_rho_mse = [
        0.004288101770848347,
        0.0007244288936820754,
        0.0005287872013394197,
        8.860566776746858e-05,
        4.576496524813782e-05,
    ]
    MLE_direct_mu_mse = [
        0.1442353669335497,
        0.026528609513416915,
        0.01173022658373048,
        0.001743117480995894,
        0.0011716604691782366,
    ]
    MLE_direct_rho_mse = [
        0.004288204690560043,
        0.0007243983407541394,
        0.0005287484357135348,
        8.860330125507665e-05,
        4.576894377374339e-05,
    ]
    W2_method1_mu_mse = [
        0.4523916507403213,
        0.09204402913469771,
        0.03735916823938818,
        0.007218564632202504,
        0.003545371611991415,
    ]
    W2_method1_rho_mse = [
        0.01257138308162584,
        0.002552666038420643,
        0.0019932433174112084,
        0.0008658928083427117,
        0.0004030746187678101,
    ]
    W1_method2_mu_mse = [
        0.14213415096728715,
        0.025844082005507415,
        0.01235892388975732,
        0.0018461559814591034,
        0.001257062668380139,
    ]
    W1_method2_rho_mse = [
        0.004662087965978105,
        0.0007260706985304167,
        0.0005548000498658098,
        9.805050618082715e-05,
        4.3803321355523233e-05,
    ]
    W2_method3_mu_mse = [
        0.1368572333359839,
        0.02573960599438422,
        0.01230802552242899,
        0.0018924838313309808,
        0.0011988902290235711,
    ]
    W2_method3_rho_mse = [
        0.0044087461042990024,
        0.0007638140367890381,
        0.0005616888468380766,
        9.47864175781428e-05,
        4.4157577900575115e-05,
    ]
    W1_method3_mu_mse = [
        0.13411514761562088,
        0.025677599696025635,
        0.012401470728391198,
        0.0018526873398687872,
        0.0012607108210958935,
    ]
    W1_method3_rho_mse = [
        0.004434107466124998,
        0.0007290924865208192,
        0.0005547070426287908,
        9.772898891301804e-05,
        4.378383900987663e-05,
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
