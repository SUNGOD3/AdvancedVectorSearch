#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Clean up
echo "Cleaning up..."
python src/setup.py clean --all
rm -rf build/
rm -f src/advanced_search_cpp*.so


# 安裝 Python 依賴項
# echo "Installing Python dependencies..."
# python -m pip install --upgrade pip
# pip install -r requirements.txt
# pip install pybind11 numpy

# 安裝系統依賴項
# echo "Installing system dependencies..."
# sudo apt-get update
# sudo apt-get install -y cmake

# 編譯 C++ 擴展
echo "Compiling C++ extension..."
python src/setup.py build_ext --inplace

# 運行測試
echo "Running tests..."
python -m unittest discover tests

# 運行基準測試
echo "Running benchmarks..."
python benchmarks/run_benchmarks.py
python benchmarks/plot.py

echo "All tasks completed successfully."