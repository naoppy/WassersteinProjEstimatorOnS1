import datetime
from typing import Optional

import matplotlib.pylab as plt
import numpy as np
import ot
from scipy.special import iv

"""
このファイルは、2つのフォンミーゼス分布のサンプル間のWasserstein距離が
p=1, 2, 8など変えた時に輸送がどのように変化するか、どのくらい変わるかを実験したときのものです。
兎に角輸送が変化することはわかりました。
"""


def pdf_von_Mises(theta, mu, kappa):
    pdf = np.exp(kappa * np.cos(theta - mu)) / (2.0 * np.pi * iv(0, kappa))
    return pdf


def print_result(cost, log_dict, p: int):
    print(
        f"p: {p}, 100*cost: {100*cost[0]:.6f}, 100*theta: {100*log_dict['optimal_theta'][0]:.3f}"
    )
    if len(log_dict) != 1:
        print(log_dict)


def confirm(empty_as: Optional[bool] = None) -> bool:
    """ファイルの保存確認を表示して、入力を受け付ける。

    Args:
        empty_as (Optional[bool], optional): 入力無しのエンターをどう扱うか. Defaults to None.

    Returns:
        bool: 保存するかどうか
    """

    dic = {"y": True, "yes": True, "n": False, "no": False}
    if empty_as is not None:
        if empty_as:
            dic[""] = True
        else:
            dic[""] = False
    while True:
        input_str = input("保存しますか? [Y]es/[N]o >> ").lower()
        if input_str in dic:
            return dic[input_str]
        else:
            print("Error! Input again.")


def main():
    t = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    n = 1000
    mu1 = 0.1
    kappa1 = 20
    mu2 = 0
    kappa2 = kappa1
    x1 = np.random.vonmises(mu1, kappa1, size=n) + np.pi  # 平均がpiずれるので注意
    x2 = np.random.vonmises(mu2, kappa2, size=n) + np.pi
    plt.figure()
    plt.plot(np.cos(t), np.sin(t), c="k")
    plt.xlim(-1.25, 1.25)
    plt.ylim(-1.25, 1.25)
    plt.scatter(np.cos(x1), np.sin(x1), c="b")
    plt.scatter(np.cos(x2), np.sin(x2), c="r")

    cost1, log_dict1 = ot.binary_search_circle(x1, x2, p=1, log=True)
    cost2, log_dict2 = ot.binary_search_circle(x1, x2, p=2, log=True)
    cost3, log_dict3 = ot.binary_search_circle(x1, x2, p=3, log=True)
    print(f"n = {n}")
    print_result(cost1, log_dict1, p=1)
    print_result(cost2, log_dict2, p=2)
    print_result(cost3, log_dict3, p=3)

    # print if you want to see the circular distribution
    plt.show()

    save = confirm(empty_as=False)
    if save:
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
        now_txt = now.strftime("%Y%m%d%H%M%S")
        np.save(f"./data/{now_txt}_x1.npy", x1)
        np.save(f"./data/{now_txt}_x2.npy", x2)


if __name__ == "__main__":
    main()
