""""
対応するファイル: ./data/実験7_再実験.txt
"""

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def main():
    NUM_PARAMS = 2  # (mu, rho)
    param_names = ["mu", "rho"]
    TRUE_PARAMS = [np.pi / 8, 0.7]
    assert NUM_PARAMS == len(param_names) and NUM_PARAMS == len(TRUE_PARAMS)

    Ns = []
    MSE_dicts: Dict[str, Dict[int, List[float]]] = (
        dict()
    )  # [method][N] = [MSE of param1, MSE of param2, ...]
    methods = []

    with open("./data/実験7_再実験.txt", encoding="utf-8") as f:
        lines = f.readlines()
        scan_line = 0
        while scan_line < len(lines):
            if lines[scan_line].startswith("N="):
                N = int(lines[scan_line].split("=")[1])
                Ns.append(N)
                scan_line += 2
                while scan_line < len(lines) and not lines[scan_line].startswith("N="):
                    line = lines[scan_line]
                    method = line.split(":")[0]
                    if method not in methods:
                        methods.append(method)
                    MSEs = line.split(":")[1].split(", ")
                    if method not in MSE_dicts:
                        MSE_dicts[method] = dict()
                    MSE_dicts[method][N] = list(
                        map(lambda s: float(s.split("=")[1]), MSEs)
                    )
                    scan_line += 1
            else:
                scan_line += 1

    for i in range(NUM_PARAMS):  # パラメータの数
        plt.figure()
        for method in methods:  # 手法の数 (MLE1, MLE2, method1, method2, method3)
            MSEs = [MSE_dicts[method][N][i] for N in Ns]
            print(MSEs)
            plt.plot(np.log10(Ns), np.log10(MSEs), label=method, marker="o")
        plt.legend()
        plt.xlabel("log10(N)")
        plt.ylabel("log10(MSE) of " + param_names[i])
        plt.title(
            "True Parameter: "
            + ", ".join(
                [
                    "{}={}".format(param_names[j], TRUE_PARAMS[j])
                    for j in range(NUM_PARAMS)
                ]
            )
        )
        plt.show()


if __name__ == "__main__":
    main()
