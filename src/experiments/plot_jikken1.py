import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
実験1のグラフ作成用
"""


def main():
    input_csv = pd.read_csv("./data/実験1.csv")
    print(input_csv)
    input_csv["N_log10"] = input_csv["N"].apply(np.log10)

    mu_actual = 0.3
    kappa_actual = 2

    plt.figure()
    plt.xlabel("log10 N")
    plt.ylabel("abs mu error")
    mu_MLE = input_csv["mu_MLE"]
    mu_W2 = input_csv["mu_W2"]
    input_csv["abs_mu_error_MLE"] = abs(mu_MLE - mu_actual)
    input_csv["abs_mu_error_W2"] = abs(mu_W2 - mu_actual)
    plt.plot(
        input_csv["N_log10"],
        input_csv["abs_mu_error_MLE"],
        label="MLE",
        marker="o",
    )
    plt.plot(
        input_csv["N_log10"],
        input_csv["abs_mu_error_W2"],
        label="W2-estimator",
        marker="x",
    )
    plt.legend()
    plt.show()

    plt.figure()
    plt.xlabel("log10 N")
    plt.ylabel("abs kappa error")
    kappa_MLE = input_csv["kappa_MLE"]
    kappa_W2 = input_csv["kappa_W2"]
    input_csv["abs_kappa_error_MLE"] = abs(kappa_MLE - kappa_actual)
    input_csv["abs_kappa_error_W2"] = abs(kappa_W2 - kappa_actual)
    print(input_csv)
    plt.plot(
        input_csv["N_log10"],
        input_csv["abs_kappa_error_MLE"],
        label="MLE",
        marker="o",
    )
    plt.plot(
        input_csv["N_log10"],
        input_csv["abs_kappa_error_W2"],
        label="W2-estimator",
        marker="x",
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
