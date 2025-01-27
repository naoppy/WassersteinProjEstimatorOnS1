import matplotlib.pyplot as plt
import numpy as np


def main():
    true_mu = np.pi / 2
    true_rho = 0.2
    N = [100, 500, 1000, 5000, 10000]
    MLE_mu_mse = [
        0.15001680206908063,
        0.022772136487325515,
        0.012668486445570847,
        0.0030595046049261983,
        0.001257543463085998,
    ]
    MLE_rho_mse = [
        0.005539324319635751,
        0.0011311005189888832,
        0.0004936637305302218,
        0.00011250200455924422,
        5.605918182331259e-05,
    ]
    method1_mu_mse = [
        2.338188449675586,
        0.6604557173825847,
        0.640369265583209,
        0.48652223628445157,
        0.5604967125900427,
    ]
    method1_rho_mse = [
        0.10665171206202201,
        0.18165053313441146,
        0.21369652525745422,
        0.2869345789952625,
        0.29045586447490046,
    ]
    method2_mu_mse = [
        0.140499046161103,
        0.018676201008639176,
        0.012020750712154542,
        0.0027305780536213526,
        0.0011189067765339138,
    ]
    method2_rho_mse = [
        0.004779351631700877,
        0.0009801786515921463,
        0.0004188392056146227,
        9.509327638479583e-05,
        4.860320004564109e-05,
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
