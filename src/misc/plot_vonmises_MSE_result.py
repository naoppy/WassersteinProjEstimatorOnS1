import matplotlib.pyplot as plt
import numpy as np


def main():
    true_mu = 0.3
    true_kappa = 2
    N = [100, 500, 1000, 5000, 10000]
    MLE_mu_mse = [
        0.006892402997718764,
        0.0013820937264447072,
        0.0007295197195663078,
        0.0001486803634282803,
        7.066909321807326e-05,
    ]
    MLE_kappa_mse = [
        0.08238419495008367,
        0.015298057754109569,
        0.005334823471411806,
        0.0009982450381136388,
        0.000701949724550066,
    ]
    method1_mu_mse = [
        0.017562137282321852,
        0.0033145317818532626,
        0.0019230730511849289,
        0.0003694393209940021,
        0.0002562707138927864,
    ]
    method1_kappa_mse = [
        0.7789677346194458,
        0.1333664492304043,
        0.0795149174761753,
        0.04677282234400227,
        0.04373348238534207,
    ]
    method2_mu_mse = [
        0.006840084547359661,
        0.0014216057745092762,
        0.0007663894955639048,
        0.0001478161968785345,
        7.2236688817582e-05,
    ]
    method2_kappa_mse = [
        0.08546328892275554,
        0.016808974849090597,
        0.005104221059516131,
        0.0010758501326452756,
        0.0006832398768744339,
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
