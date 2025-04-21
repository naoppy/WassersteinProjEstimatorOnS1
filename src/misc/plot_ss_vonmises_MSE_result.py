import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd


def main():
    N_METHOD = 3
    METHOD_NAME = ["MLE", "W2(method1)", "W1(method2)"]
    N_PARAM = 3
    PARAM_NAME = ["mu", "kappa", "lambda"]
    N_KINDS = N_METHOD * N_PARAM

    def getIdx(method: int, param: int) -> int:
        return method * N_PARAM + param

    def getName(idx: int) -> str:
        method = idx // N_PARAM
        param = idx % N_PARAM
        return f"{METHOD_NAME[method]}_{PARAM_NAME[param]}"

    N_list = []
    KIND_list = [getName(i) for i in range(N_KINDS)]
    data_mat = []

    with open("data/実験9.txt", "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("N="):
            N = int(lines[i].split("=")[1])
            tmp = []
            N_list.append(N)
            i += 1
            for _ in range(N_METHOD):
                i += 2
                nums = lines[i].split(", ")
                for param in range(N_PARAM):
                    tmp.append(float(nums[param]))
            data_mat.append(tmp)
        else:
            i += 1
    df = pd.DataFrame(data_mat, columns=KIND_list, index=N_list)
    print(df)
    df.to_csv("data/実験9.csv")


if __name__ == "__main__":
    main()
