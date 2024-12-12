# AdvancedVectorSearch ![Version](https://img.shields.io/github/v/tag/SUNGOD3/AdvancedVectorSearch?label=version)

[![codecov](https://codecov.io/gh/SUNGOD3/AdvancedVectorSearch/graph/badge.svg?token=ZXNRPXKMGE)](https://codecov.io/gh/SUNGOD3/AdvancedVectorSearch)
[![CI](https://github.com/SUNGOD3/AdvancedVectorSearch/actions/workflows/ci.yml/badge.svg)](https://github.com/SUNGOD3/AdvancedVectorSearch/actions/workflows/ci.yml)
![Languages](https://img.shields.io/github/languages/top/SUNGOD3/AdvancedVectorSearch)
[![Google Slides](https://img.shields.io/badge/Google%20Slides-Presentation-blue?logo=google-slides&logoColor=white)](https://docs.google.com/presentation/d/1kZTVmiuk8j7mkV637VwyCPt_E7LSCqHtCFL07N9o3pg/edit?usp=sharing)

This project aims to develop a high-speed and high-accuracy similarity vector search method.

## Problem to Solve

1. Improving search speed beyond basic linear search, especially for high-dimensional vector spaces common in RAG systems.
2. Developing new similarity metrics that may be more suitable for specific RAG applications.
3. Creating a flexible, extensible framework that allows for easy experimentation with different search algorithms and metrics.
4. Balancing speed, accuracy, and memory usage in vector search operations.

## System Architecture
### Hardware:
* CPU: 4 cores (recommended for smooth operation)
* Memory: 4 GB of RAM
* Storage: 32 GB of available SSD space
### Software:
* Operating System: Ubuntu 20.04 or later (used in CI/CD pipeline)
* Python: Version 3.8 or later
* C++ Compiler: C++17 compatible (required for compiling extensions)
* Build Tools:
  * cmake (for building C++ extensions)
  * pip (Python package manager)
* Python Dependencies:
  * pybind11
  * numpy
  * Other dependencies listed in requirements.txt
 
## How to build?
```bash
git clone https://github.com/SUNGOD3/AdvancedVectorSearch.git
cd AdvancedVectorSearch

#bash
echo "Installing Python dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y cmake

echo "Compiling C++ extension..."
python src/setup.py build_ext --inplace
```

