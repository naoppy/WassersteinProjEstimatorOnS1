import numpy as np


def _main():
    roll_num = 4
    y = np.roll(np.arange(10), roll_num)
    print(y)
    i = 0
    if y[-1] <= y[0]:  # y2[i-1] <= y[i]
        i += 1
        while y[i - 1] <= y[i]:
            i += 1
    # 既にソート済みならi=0, 全て同じ値(pdfがデルタ関数)ならi=sample_num, それ以外ならiは変曲点の奥のidx
    z = np.roll(y, -i)
    print(i)
    print(z)


if __name__ == "__main__":
    _main()
