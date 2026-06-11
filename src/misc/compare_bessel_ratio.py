"""
変形ベッセル関数の比 I_1(kappa) / I_0(kappa) の計算手法を比較するスクリプト。
フォンミーゼス分布 (von Mises distribution) の実験で kappa が大きいとき (>=710) に
生じるオーバーフロー問題を解決するための調査目的で作成。

【検証手法】
1. Direct (i1/i0): 標準的な scipy.special.i1 / i0
2. IVE (ive1/ive0): 指数スケーリングされた scipy.special.ive を使用 (I_v(k) * e^-|k|)
3. Log-Diff: 対数の差を取ってから exp を計算 (exp(log(I_1) - log(I_0)))

【検証から分かったこと】
1. オーバーフロー回避:
   - Direct は kappa >= 710 で i0, i1 が float64 の上限を超えて inf となり、
     inf/inf の除算から nan になる。
   - IVE および Log-Diff は kappa = 100,000 のような大域でもオーバーフローせず
     正常に計算可能。
2. 精度:
   - IVE と Log-Diff の差分は kappa = 100,000 でも最大 1.5e-12 程度と極めて小さく、
     倍精度浮動小数点の精度範囲内で完全に一致している。
3. 計算速度:
   - 小さい kappa (例: 4.0) の場合、Direct が最も高速 (IVE の約 2.1 倍高速)。
   - IVE は Log-Diff に比べて約 1.6 倍高速。

【結論】
小さい kappa での高速性と、大きい kappa でのオーバーフロー耐性を両立するため、
以下のようなハイブリッド（閾値分岐）方式を採用するのが望ましい：
    if kappa < 700:
        return i1(kappa) / i0(kappa)
    else:
        return ive(1, kappa) / ive(0, kappa)
"""

import time

import numpy as np
from scipy.special import i0, i1, ive


def ratio_direct(kappa):
    """標準的な i1(kappa) / i0(kappa)"""
    try:
        return i1(kappa) / i0(kappa)
    except Exception:
        return np.nan


def ratio_ive(kappa):
    """指数スケーリングされた ive(1, kappa) / ive(0, kappa)"""
    return ive(1, kappa) / ive(0, kappa)


def ratio_log(kappa):
    """対数を取ったベッセル関数の差から計算するバージョン
    log(I_v(kappa)) = log(ive(v, kappa)) + kappa
    """
    log_i1 = np.log(ive(1, kappa)) + kappa
    log_i0 = np.log(ive(0, kappa)) + kappa
    return np.exp(log_i1 - log_i0)


def main():
    # テストする kappa の値の範囲
    kappas = [
        0.0001,
        0.01,
        0.1,
        1.0,
        4.0,
        10.0,
        50.0,
        100.0,
        500.0,
        700.0,
        710.0,
        720.0,
        730.0,
        1000.0,
        5000.0,
        10000.0,
        100000.0,
    ]

    header = (
        f"{'kappa':>10} | {'Direct (i1/i0)':>15} | "
        f"{'IVE (ive1/ive0)':>15} | {'Log-Diff (exp(log1-log0))':>25} | "
        f"{'Diff (IVE - Log)':>18}"
    )
    print(header)
    print("-" * 95)

    for k in kappas:
        r_dir = ratio_direct(k)
        r_ive = ratio_ive(k)
        r_log = ratio_log(k)

        # 差分の計算
        diff = abs(r_ive - r_log)

        row = (
            f"{k:>10.4f} | {r_dir:>15.10f} | {r_ive:>15.10f} | "
            f"{r_log:>25.10f} | {diff:>18.10e}"
        )
        print(row)

    print("\n[Performance test] Running 1,000,000 iterations for kappa=4.0...")

    # Direct (try-exceptガードあり) の時間計測
    start = time.time()
    for _ in range(1000000):
        _ = ratio_direct(4.0)
    time_direct = time.time() - start

    # Direct (ガードなし) の時間計測
    start = time.time()
    for _ in range(1000000):
        _ = i1(4.0) / i0(4.0)
    time_direct_pure = time.time() - start

    # IVE の時間計測
    start = time.time()
    for _ in range(1000000):
        _ = ratio_ive(4.0)
    time_ive = time.time() - start

    # Log の時間計測
    start = time.time()
    for _ in range(1000000):
        _ = ratio_log(4.0)
    time_log = time.time() - start

    print(f"Direct (with try-except): {time_direct:.4f} seconds")
    print(f"Direct (pure i1/i0):     {time_direct_pure:.4f} seconds")
    print(
        f"IVE method (ive1/ive0):  {time_ive:.4f} seconds "
        f"(Ratio: {time_ive / time_direct_pure:.2f}x of Pure Direct)"
    )
    print(
        f"Log method:              {time_log:.4f} seconds "
        f"(Ratio: {time_log / time_ive:.2f}x of IVE)"
    )


if __name__ == "__main__":
    main()
