"""実験結果のCSVデータを簡易的に可視化するためのスクリプト。

LaTeX（TikZ）で本番用のグラフを作成する前に、手軽にデータの傾向を確認するために使用します。
- 引数なしで実行した場合、`data/` ディレクトリ内で最も更新日時が新しい
  CSV ファイルを自動検出してプロットします。
- 引数として CSV ファイルのパスを直接指定してプロットすることも可能です。
- `-i` オプションを指定すると、`data/` ディレクトリ内の CSV ファイルを
  対話的にリスト選択できます。
- 列名からパラメータ（mu, kappa, rho, lambda）や評価指標（kl, w1, w2, time）を
  自動で検出し、それぞれ別のサブプロットに自動分割してプロットします。
- データ範囲に応じて、自動的に対数スケール（log-scale）でプロットを行います。
"""

import argparse
import glob
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_latest_csv(data_dir: str = "./data") -> Optional[str]:
    """Find the most recently modified CSV file in the data directory."""
    csv_files = glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)
    if not csv_files:
        return None
    return max(csv_files, key=os.path.getmtime)


def select_csv_interactively(data_dir: str = "./data") -> Optional[str]:
    """List all CSV files in the data directory and ask the user to choose one."""
    csv_files = sorted(glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True))
    if not csv_files:
        print("No CSV files found in data directory.")
        return None

    print("\n--- Available CSV Files ---")
    for idx, path in enumerate(csv_files):
        print(f"[{idx}] {path}")
    print(f"[{len(csv_files)}] [Cancel / Exit]")

    while True:
        try:
            choice = input(
                f"Select a file [0-{len(csv_files)}] (Press Enter to exit): "
            ).strip()
            if not choice or int(choice) == len(csv_files):
                return None
            idx = int(choice)
            if 0 <= idx < len(csv_files):
                return csv_files[idx]
        except ValueError:
            pass
        print("Invalid choice. Please enter a valid index.")


def plot_csv(filepath: str) -> None:
    """Load the CSV and plot the contents, auto-separating columns into subplots."""
    df = pd.read_csv(filepath, index_col=0)
    print(f"\nShowing: {filepath}")
    print(df.head())

    # Classification keywords
    param_kws = ["mu", "kappa", "rho", "lambda"]
    metric_kws = ["kl", "w1", "w2", "time"]

    groups: Dict[str, List[str]] = {kw: [] for kw in param_kws + metric_kws}
    unclassified: List[str] = []

    # Two-pass classification: parameter names take priority
    for col in df.columns:
        classified = False
        for kw in param_kws:
            if kw in col.lower():
                groups[kw].append(col)
                classified = True
                break
        if classified:
            continue

        for kw in metric_kws:
            if kw in col.lower():
                groups[kw].append(col)
                classified = True
                break
        if not classified:
            unclassified.append(col)

    # Gather active groups that actually have columns
    active_groups = {kw: cols for kw, cols in groups.items() if cols}
    if unclassified:
        active_groups["other"] = unclassified

    x_label = df.index.name or "N"
    x_values = df.index.to_numpy()

    # Determine if X-axis should be log scale (e.g. N = [100, 1000, 10000...])
    use_log_x = False
    try:
        if (
            len(x_values) > 1
            and np.all(x_values > 0)
            and (np.max(x_values) / np.min(x_values) >= 10)
        ):
            use_log_x = True
    except Exception:
        pass

    if not active_groups:
        # Fallback: Plot all in a single plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for col in df.columns:
            ax.plot(df.index, df[col], label=col, marker="o")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Value")
        if use_log_x:
            ax.set_xscale("log")
        ax.set_title(f"Visualization: {os.path.basename(filepath)}")
        ax.legend()
        ax.grid(True)
    else:
        # Create side-by-side subplots for each classified parameter/metric group
        n_plots = len(active_groups)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5.5), sharex=True)
        if n_plots == 1:
            axes = [axes]

        for i, (kw, cols) in enumerate(active_groups.items()):
            ax = axes[i]
            for col in cols:
                ax.plot(df.index, df[col], label=col, marker="o")

            ax.set_xlabel(x_label)
            if use_log_x:
                ax.set_xscale("log")

            # Determine Y-axis log scale dynamically
            # (Only if all values are strictly positive and cover a wide range)
            try:
                all_pos = True
                for col in cols:
                    if not (df[col] > 0).all():
                        all_pos = False
                        break
                if all_pos:
                    y_min = df[cols].min().min()
                    y_max = df[cols].max().max()
                    if y_min > 0 and (y_max / y_min >= 20):
                        ax.set_yscale("log")
            except Exception:
                pass

            ax.set_ylabel("log10(MSE) or Value")
            ax.set_title(f"Metric/Parameter: {kw.upper()}")
            ax.legend(fontsize=8, loc="best")
            ax.grid(True)

        fig.suptitle(f"Quick Visualization: {os.path.basename(filepath)}", fontsize=12)

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quickly visualize experimental CSV results."
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Path to the CSV file. If omitted, shows options.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Interactively select from all available CSV files.",
    )
    args = parser.parse_args()

    target_file = None

    if args.file:
        target_file = args.file
    elif args.interactive:
        target_file = select_csv_interactively()
    else:
        # Default behavior: auto-detect the latest modified file
        target_file = find_latest_csv()
        if target_file:
            print(f"Auto-detected latest result: {target_file}")
        else:
            print("No CSV files found in './data'. Please specify a file path.")
            return

    if target_file:
        if os.path.exists(target_file):
            plot_csv(target_file)
        else:
            print(f"Error: File '{target_file}' does not exist.")


if __name__ == "__main__":
    main()
