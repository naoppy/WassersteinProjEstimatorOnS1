import matplotlib.pyplot as plt
import numpy as np


def main():
    true_mu = np.pi / 2
    true_rho = 0.7
    N = [100, 500, 1000, 5000, 10000]
    MLE_mu_mse = [
        0.020969880041902585,
        0.005395466952956968,
        0.002309479208641108,
        0.0005181673031947092,
        0.00025004340516233774,
    ]
    MLE_rho_mse = [
        0.010635299197330562,
        0.0020941922473382223,
        0.001129026772362307,
        0.0002021788745164216,
        0.00010828175301010574,
    ]
    method1_mu_mse = [
        0.5185261356414551,
        0.5482008002519084,
        0.5593507415300156,
        0.5493448381139854,
        0.5541964990114009,
    ]
    method1_rho_mse = [
        0.05786279239165201,
        0.0602864319194613,
        0.058784654820370756,
        0.06241905030221526,
        0.06067076591809501,
    ]
    method2_mu_mse = [
        0.003404435781272154,
        0.0009203494646046281,
        0.0003117486171322636,
        6.950051607217104e-05,
        2.854162737154224e-05,
    ]
    method2_rho_mse = [
        0.0018628021691602641,
        0.0004027284605713283,
        0.00016289890432216625,
        3.7324961550192025e-05,
        1.2873101016795889e-05,
    ]
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
        np.log10(method2_mu_mse),
        label="W1-estimator from method2",
        marker="s",
        linestyle="dashed",
    )
    plt.xlabel("log10(N)")
    plt.ylabel("log10(MSE) of mu")
    plt.legend()
    plt.title(f"True parameter: mu={true_mu}, rho={true_rho}")
    plt.show()

    plt.plot(
        np.log10(N),
        np.log10(MLE_rho_mse),
        label="MLE",
        marker="o",
        linestyle="dashdot",
    )
    plt.plot(
        np.log10(N),
        np.log10(method1_rho_mse),
        label="W2-estimator from method1",
        marker="x",
        linestyle="solid",
    )
    plt.plot(
        np.log10(N),
        np.log10(method2_rho_mse),
        label="W1-estimator from method2",
        marker="s",
        linestyle="dashed",
    )
    plt.xlabel("log10(N)")
    plt.ylabel("log10(MSE) of rho")
    plt.legend()
    plt.title(f"True parameter: mu={true_mu}, rho={true_rho}")
    plt.show()


if __name__ == "__main__":
    main()
