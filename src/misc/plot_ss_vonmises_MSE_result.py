
from ..plots import txt2csv


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
