from typing import List

import numpy as np
import pandas as pd


def ToCSV(
    METHOD_NAME: List[str],
    PARAM_NAME: List[str],
    input_filename: str,
    to_filename: str,
    fisher_mat_inv_diag_list: List[float],
) -> None:
    N_METHOD = len(METHOD_NAME)
    N_PARAM = len(PARAM_NAME)
    N_KINDS = N_METHOD * N_PARAM

    def getIdx(method: int, param: int) -> int:
        return method * N_PARAM + param

    def getName(idx: int) -> str:
        method = idx // N_PARAM
        param = idx % N_PARAM
        return f"{METHOD_NAME[method]}_{PARAM_NAME[param]}"

    log10_N_list = []
    KIND_list = [getName(i) for i in range(N_KINDS)]
    data_mat = []

    with open(input_filename, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("N="):
            N = int(lines[i].split("=")[1])
            tmp = []
            log10_N_list.append(np.log10(N))
            i += 1
            for _ in range(N_METHOD):
                i += 2
                nums = lines[i].split(", ")
                for param in range(N_PARAM):
                    tmp.append(np.log10(float(nums[param])))
            data_mat.append(tmp)
        else:
            i += 1
    df = pd.DataFrame(data_mat, columns=KIND_list, index=log10_N_list)
    for i, param in enumerate(PARAM_NAME):
        df["Cramer-Rao Lower Bound of " + param] = (
            -np.log10(fisher_mat_inv_diag_list[i]) - log10_N_list
        )
    print(df)
    df.to_csv(to_filename)
