"""
簡単にCSVのデータを可視化する用
特に綺麗に表示することにはこだわらない
"""

import matplotlib.pyplot as plt
import pandas as pd


def _main():
    # CSVファイルの読み込み
    df = pd.read_csv(
        "./data/csv_data/vonmises_MSE_cond1/ex6_mu0.3kappa2.csv",
        # "./data/csv_data/wrapcauchy_MSE_cond1/ex7.csv",
        index_col=0,
    )

    # データの表示
    print(df)

    # グラフの描画
    plt.figure(figsize=(10, 6))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)
    plt.xlabel("log10(N)")
    plt.ylabel("log10(MSE)")
    plt.title("CSV Data Visualization")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    _main()
