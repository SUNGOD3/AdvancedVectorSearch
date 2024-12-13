#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Clean up
echo "Cleaning up..."
python src/setup.py clean --all
rm -rf build/
rm -f src/advanced_search_cpp*.so


# install python dependencies
# echo "Installing Python dependencies..."
# python -m pip install --upgrade pip
# pip install -r requirements.txt
# pip install pybind11 numpy

# install system dependencies
# echo "Installing system dependencies..."
# sudo apt-get update
# sudo apt-get install -y cmake

# Compile C++ extension
echo "Compiling C++ extension..."
cd src
python setup.py build_ext --inplace --build-lib=../
cd ..

# run python tests
echo "Running Python tests..."
python -m unittest discover tests

# run C++ tests
echo "Running C++ tests..."
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
ctest
lcov --capture --directory . --output-file coverage.info
lcov --remove coverage.info '/usr/*' --output-file coverage.info
lcov --list coverage.info
cd ..

# run benchmarks
echo "Running benchmarks..."
python benchmarks/run_benchmarks.py
python benchmarks/plot.py

echo "All tasks completed successfully."