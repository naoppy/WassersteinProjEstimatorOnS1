#!/bin/bash

# シェルスクリプト実行時にエラーで止める
set -e

# Poetry仮想環境で実行する python -m モジュールの一覧
modules=(
  src.experiments.ex6_vonmises_MSE
)

# 各モジュールを順番に実行
for module in "${modules[@]}"
do
  echo "Running: python -m $module"
  poetry run python -m "$module"
done
