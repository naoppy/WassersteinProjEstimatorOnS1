"""
簡単にCSVのデータを可視化する用
特に綺麗に表示することにはこだわらない
"""

import matplotlib.pyplot as plt
import pandas as pd


def _main():
    # CSVファイルの読み込み
    df = pd.read_csv(
        # "./data/csv_data/vonmises_MSE/ex6_mu0.3kappa2.csv",
        "./data/csv_data/vonmises_MSE/ex65_MSE_kappa_change.csv",
        # "./data/csv_data/wrapcauchy_MSE/ex7.csv",
        # "./data/csv_data/vonmises_mix/ex8_vonmises_mix_MSE.csv",
        index_col=0,
    )
    xylabel = ["log10(N)", "log10(MSE)"]
    # xylabel = ["kappa", "MSE/CR"]

    # データの表示
    print(df)

    # グラフの描画
    plt.figure(figsize=(10, 6))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)
    plt.xlabel(xylabel[0])
    plt.ylabel(xylabel[1])
    plt.title("CSV Data Visualization")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    _main()
