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
The system is designed to integrate Python and C++ seamlessly, leveraging the computational efficiency of C++ and the flexibility of Python. The architecture primarily revolves around advanced search functionalities, including linear search and KNN (k-nearest neighbor) search implemented with ball trees. Pybind11 acts as the bridging tool, allowing C++ modules to be directly accessible in Python.

### Modules and Responsibilities

#### Python Components

* **search.py:**
  
  * Provides high-level interfaces for advanced search methods, including AdvancedLinearSearch and AdvancedKNNSearch.

  * Integrates with FAISS for additional support in HNSW (Hierarchical Navigable Small World) searches.

#### C++ Components

* **BaseAdvancedSearch:**

  * Serves as the abstract base class for all advanced search implementations.

* **AdvancedLinearSearch:**

  * Implements efficient linear search functionality.

  * Includes optimizations for vector normalization and distance computation.

* **AdvancedKNNSearch:**

  * Implements KNN search using ball trees for efficient nearest neighbor queries.

  * Includes mechanisms for constructing and querying the ball tree.

### Integration Using Pybind11

The integration layer is built using pybind11, which allows seamless exposure of C++ classes and functions to Python. Key aspects of the integration include:

* Data Interoperability: Numpy arrays in Python are directly mapped to std::vector or Eigen objects in C++ using pybind11's utilities like py::array_t.

* Class Binding: C++ classes such as AdvancedLinearSearch and AdvancedKNNSearch are bound to Python, enabling object-oriented interaction from Python scripts.

* Error Handling: Custom exceptions in C++ are translated to Python exceptions, ensuring consistency in debugging.

## Index Module

The Index Module provides various search algorithms for efficiently retrieving the most similar vectors from a collection. It supports different distance metrics and uses both Python and C++ implementations for performance optimization. The module consists of the following classes:

### 1. **AdvancedLinearSearch**
This class implements a linear search algorithm, which finds the nearest neighbors based on a chosen distance metric. It uses a C++ extension for performance optimization.

- **Constructor Parameters:**
  - `vectors` (np.ndarray): A 2D array of vectors to search through.
  - `metric` (str): The distance metric to use. Options include `"cosine"`, `"l2"`, and `"inner_product"`. The default is `"cosine"`.
  
- **Methods:**
  - `search(query, k)`:
    - `query` (np.ndarray): The query vector.
    - `k` (int): The number of nearest neighbors to return.
    - Returns the indices of the `k` most similar vectors.

### 2. **AdvancedKNNSearch**
This class performs a K-Nearest Neighbors (KNN) search using a C++ implementation for efficient computation.

- **Constructor Parameters:**
  - `vectors` (np.ndarray): A 2D array of vectors to search through.
  - `metric` (str): The distance metric to use. Options are `"cosine"`, `"l2"`, and `"inner_product"`. The default is `"cosine"`.
  
- **Methods:**
  - `search(query, k)`:
    - `query` (np.ndarray): The query vector.
    - `k` (int): The number of nearest neighbors to return.
    - Returns the indices of the `k` most similar vectors.

### 3. **AdvancedHNSWSearch**
This class leverages Faiss' HNSW (Hierarchical Navigable Small World) algorithm for high-performance vector search with enhanced accuracy. It supports multiple distance metrics and allows fine-tuning of the search parameters.

- **Constructor Parameters:**
  - `vectors` (np.ndarray): A 2D array of vectors to search through.
  - `metric` (str): The distance metric to use. Options include `"cosine"`, `"l2"`, and `"inner_product"`. The default is `"cosine"`.
  - `ef_construction` (int): The depth of layer construction. A higher value increases accuracy but decreases performance. Default is `250`.
  - `M` (int): The maximum number of connections per element. Default is `64`.
  - `ef_search` (int, optional): The effective search parameter, which controls the search accuracy and performance. If not provided, it is set dynamically based on the dataset size.

- **Methods:**
  - `search(query, k)`:
    - `query` (np.ndarray): The query vector.
    - `k` (int): The number of nearest neighbors to return.
    - Returns the indices of the `k` most similar vectors.

### 4. **LinearSearch**
This class implements a straightforward brute-force search algorithm. It computes distances between a query vector and all other vectors in the dataset using the selected distance metric.

- **Constructor Parameters:**
  - `vectors` (np.ndarray): A 2D array of vectors to search through.
  - `metric` (str): The distance metric to use. Options include `"cosine"`, `"l2"`, and `"inner_product"`. The default is `"cosine"`.

- **Methods:**
  - `search(query, k)`:
    - `query` (np.ndarray): The query vector.
    - `k` (int): The number of nearest neighbors to return.
    - Returns the indices of the `k` most similar vectors.

### 5. **FaissSearch**
This class uses Faiss to perform similarity searches, supporting multiple distance metrics. It normalizes vectors for cosine and inner product metrics, ensuring consistent performance.

- **Constructor Parameters:**
  - `vectors` (np.ndarray): A 2D array of vectors to search through.
  - `metric` (str): The distance metric to use. Options include `"cosine"`, `"l2"`, and `"inner_product"`. The default is `"cosine"`.

- **Methods:**
  - `search(query, k)`:
    - `query` (np.ndarray): The query vector.
    - `k` (int): The number of nearest neighbors to return.
    - Returns the indices of the `k` most similar vectors.

### Notes
- The module does **not** provide functionality for adding or removing vectors from the index after initialization. The vectors used for indexing must be supplied when the index is created.
- **HNSW search accuracy**: The accuracy of the AdvancedHNSWSearch method may not be 100% when dealing with very large datasets, as it sacrifices some precision for faster performance. In contrast, other search methods (such as linear search and KNN) perform strict comparisons and guarantee 100% accuracy.

## Requirements

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

