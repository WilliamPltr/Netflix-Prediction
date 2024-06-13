from Modulew import Matrix
from Module_Sparse import SparseMatrixCSR
from Module_Sparse import SparseMatrixTUPLES
import sys
import numpy as np
import time
from scipy import sparse
from scipy.sparse import csr_matrix
from PlotSVDBeforeAndAfterReduction import perform_and_plot_svd
from PartTwoPlotOnFasterScipyNumpy import compare_svd_methods
from SparseCSRExecutionSPEED import PlotCSR
from SparseTUPLEExecutionSPEED import PlotTuple

A_data = np.random.randint(6, size=(30,30))
B_data = np.random.randint(6, size=(30, 30))
C_vector_data = np.random.randint(6, size=(30, 1))
D_sparse_matrix = sparse.random(30,30, density=0.2, format='csr')
D_t = D_sparse_matrix.tocsr()
E_sparse_matrix = sparse.random(30,30, density=0.2, format='csr')
E_t = E_sparse_matrix.tocsr()

A = Matrix((A_data))
B = Matrix((B_data))
C = np.array(C_vector_data)
D = SparseMatrixCSR(D_t)
E = SparseMatrixCSR(E_t)

Domain = sys.argv[1]
Operation = sys.argv[2]

if Domain == 'Matrix' or Domain == 'DenseMatrix':
    if Operation == 'A':
        print("\nA:")
        print(A.matrix)
    elif Operation == 'B':
        print("\nB:")
        print(B.matrix)
    elif Operation == 'add':
        add_result = A.add(B).matrix
        print("\nA + B:")
        print(add_result)
    elif Operation == 'sub':
        subtract_result = A.sub(B).matrix
        print("\nA - B:")
        print(subtract_result)
    elif Operation == 'scalar_multiply':
        scalar_multiply_result = A.scalar_multiply(2).matrix
        print("\nA * 2:")
        print(scalar_multiply_result)
    elif Operation == 'dot_product':
        dot_product = A.multiply(B).matrix
        print("\nA * B:")
        print(dot_product)
    elif Operation == 'matrix_dot_vector':
        matrix_vector = A.matrix_x_vector(C)
        print("\nA * C:")
        print(matrix_vector)
    elif Operation == 'transpose':
        transposed = A.transpose().matrix
        print("\nTranspose of A:")
        print(transposed)
    elif Operation == 'L1_norm':
        L1 = A.norm_L1()
        print("\nNorm L1 of A:")
        print(L1)
    elif Operation == 'L2_norm':
        L2 = A.norm_L2()
        print("\nNorm L2 of A:")
        print(L2)
    elif Operation == 'Linf_norm':
        Linf = A.norm_Linf()
        print("\nNorm Linf of A:")
        print(Linf)
    elif Operation == 'eigenvalues':
        Eigenvalues = A.eigenvalues()
        print("\nEigenvalues of A:")
        print(Eigenvalues)
    elif Operation == 'SVD':
        U,S,V = A.svd_compute()
        print("\nMatrix U of A:")
        print(U)
        print("\nSingular Values of A:")
        print(S)
        print("\nMatrix V of A:")
        print(V)
    elif Operation == 'solve_linear_system':
        solutions = A.solve_dense(C)
        print("\nSolutions of Ax = C:")
        print(solutions)
    else:
        print('Tap a correct operation')

if Domain == 'SparseMatrix':
    if Operation == 'D':
        print("\nD:")
        print(D_t)
    elif Operation == 'E':
        print("\nE:")
        print(E_t)
    elif Operation == 'add':
        add_result = D.add(E)
        print("\nD + E:")
        print(add_result)
    elif Operation == 'sub':
        subtract_result = D.sub(E)
        print("\nD - E:")
        print(subtract_result)
    elif Operation == 'dot_product':
        dot_product = D.matrix_x_matrix(E)
        print("\nD * E:")
        print(dot_product)
    elif Operation == 'matrix_dot_vector':
        matrix_vector = D.matrix_x_vector(C)
        print("\nD * C:")
        print(matrix_vector)
    elif Operation == 'transpose':
        transposed = D.transpose()
        print("\nTranspose of D:")
        print(transposed)
    elif Operation == 'sparse_to_dense':
        dense_repr = D.to_dense()
        print("\nNormal representation of D:")
        print(dense_repr)
    else:
        print('Tap a correct operation')

if Domain == 'MovieRecommendation':
    Z = Matrix.generate_random_matrix(1000,500)
    if Operation == 'Transform_to_binary':
        binary_matrix = Matrix(Z.transform_to_binary())
        print("\nAssociated binary matrix of a random matrix:")
        print(binary_matrix.matrix)
    elif Operation == 'SingularValues':
        compare_svd_methods(users=100,movies=50)
    elif Operation == 'ReduceMatrix':
        U_reduced,S_reduced,Vt_reduced = Z.reduce_matrix()
        print("\nReduced U matrix:")
        print(U_reduced)
        print("\nReduced singular values:")
        print(S_reduced)
        print("\nReduced Vt matrix:")
        print(Vt_reduced)
    elif Operation == 'PlotReduction':
        perform_and_plot_svd()
    elif Operation == 'RecommendMovies':
        liked_movie_index = 2
        selected_movie_num = 3
        U_reduced,S_reduced,Vt_reduced = Z.reduce_matrix()
        recommended_movies = Z.recommend(liked_movie_index,Vt_reduced,selected_movie_num)
        print(f"\nIf you like the film type at the index {liked_movie_index}, you're going to love these {selected_movie_num} types of film !")
        print("\n",recommended_movies)
    else:
        print('Tap a correct operation')
    
if Domain == 'CSR_or_Tuples':
    if Operation == 'PlotCSR':
        PlotCSR()
    if Operation == 'PlotTuples':
        PlotTuple()

if Domain != 'Matrix' and Domain != 'DenseMatrix' and Domain != 'SparseMatrix' and Domain != 'MovieRecommendation' and Domain != 'CSR_or_Tuples':
    print('Tap a correct Domain')


