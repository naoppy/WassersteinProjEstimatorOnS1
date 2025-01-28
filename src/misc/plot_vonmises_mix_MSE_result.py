import matplotlib.pyplot as plt
import numpy as np


def main():
    true_mu = np.pi / 4
    true_kappa = 5
    uniform_noise_rate = 0.1
    N = [100, 500, 1000, 5000, 10000]
    MLE_mu_mse = [
        0.0037989348198864833,
        0.0006217868189157759,
        0.00029267969858946775,
        4.3209706544888316e-05,
        3.1853551339420144e-05,
    ]
    MLE_kappa_mse = [
        3.950711988608315,
        4.302883565244785,
        4.3079444477615665,
        4.318073881936693,
        4.340892434976493,
    ]
    method2_mu_mse = [
        0.0036554241738793584,
        0.000624615303163098,
        0.00033993615015391403,
        4.894098128606923e-05,
        3.396367380084385e-05,
    ]
    method2_kappa_mse = [
        2.144587167506339,
        1.711060408607712,
        1.669362856158109,
        1.4718892822243126,
        1.4563315588904653,
    ]
    plt.plot(
        np.log10(N), np.log10(MLE_mu_mse), label="MLE", marker="o", linestyle="dashdot"
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
    plt.title(
        f"True parameter: mu={true_mu}, kappa={true_kappa}, uniform noise rate={uniform_noise_rate}"
    )
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
        np.log10(method2_kappa_mse),
        label="W1-estimator from method2",
        marker="s",
        linestyle="dashed",
    )
    plt.xlabel("log10(N)")
    plt.ylabel("log10(MSE) of kappa")
    plt.legend()
    plt.title(
        f"True parameter: mu={true_mu}, kappa={true_kappa}, uniform noise rate={uniform_noise_rate}"
    )
    plt.show()


if __name__ == "__main__":
    main()
