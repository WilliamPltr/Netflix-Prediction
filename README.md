# 🎥 Netflix Prediction Project

### 📅 Programming For Data Science - May 2024

This guide outlines how to utilize the program, designed to perform a variety of matrix operations applicable to dense matrices, sparse matrices, and niche applications such as movie recommendation systems. The program integrates several modules; you are encouraged to modify them as necessary to ensure compatibility with your specific environment.

### 📄 Instructions

Detailed instructions for running the programs and understanding the analyses are provided in the PDF file named `data_science_instructions.pdf`. This document includes the steps to follow for the project. Please refer to this document for comprehensive guidelines.

## 🗝️ Key Components

- The user interface and the `Modulew` file, which includes the `Matrix` class equipped with essential methods, especially those relevant to the movie recommendation system, are critical components of this program.

## 🛠️ Supported Operations

### 🔹 SparseMatrix Operations:
- Addition (`add`), subtraction (`sub`), dot product (`dot_product`), matrix-vector multiplication (`matrix_dot_vector`), transpose (`transpose`), conversion from sparse to dense (`sparse_to_dense`).

### 🔹 DenseMatrix Operations:
- Includes all SparseMatrix operations, plus scalar multiplication (`scalar_multiply`), L1 norm (`L1_norm`), L2 norm (`L2_norm`), infinity norm (`Linf_norm`), eigenvalues calculation (`eigenvalues`), Singular Value Decomposition (SVD), and solving linear systems (`solve_linear_system`).

### 🔹 MovieRecommendation Operations:
- Binary matrix transformation (`Transform_to_binary`), singular values analysis (`SingularValues`), matrix reduction (`ReduceMatrix`), visualization of SVD (`PlotReduction`), and movie recommendations (`RecommendMovies`).

### 🔹 CSR_or_Tuples Operations:
- Performance plotting for CSR format (`PlotCSR`) and tuples (`PlotTuples`).

The purpose of these methods should be self-explanatory based on their names.

## 🚀 Execution

To run the program, execute it from your shell with Python, specifying the file name (here, `User_Interface`) as the first argument. The second argument should be the domain of operations you wish to perform (`SparseMatrix`, `DenseMatrix`, `MovieRecommendation`, `CSR_or_Tuples`), followed by the third argument, which is the operation you wish to execute (as detailed above).
