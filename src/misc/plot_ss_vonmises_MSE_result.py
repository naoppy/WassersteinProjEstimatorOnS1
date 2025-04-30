import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

from ..plots import txt2csv
from ..distributions import sine_skewed_vonmises


def main():
    # fisher_mat = sine_skewed_vonmises.fisher_info_3x3()
    txt2csv.ToCSV(
        ["MLE", "W2(method1)", "W1(method2)"],
        ["mu", "kappa", "lambda"],
        "data/実験9.txt",
        "data/実験9.csv",
    )


if __name__ == "__main__":
    main()
