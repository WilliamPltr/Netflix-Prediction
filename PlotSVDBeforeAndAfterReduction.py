import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def perform_and_plot_svd(matrix=None, k=None):
    
    #If no matrix is provided, use a default random binary matrix of size (1000, 2000)
    if matrix is None:
        matrix = np.random.randint(2, size=(1000, 2000))
    
    #Perform SVD on the matrix
    U, S, Vt = la.svd(matrix, full_matrices=False)
    
    #Determine the number of singular values to retain if not specified
    if k is None:
        k = 2 + 3*int(np.log(len(matrix[0])))
    
    #Ensure k does not exceed the number of singular values
    k = min(k, len(S))
    S_reduced = S[:k]
    
    plt.figure(figsize=(12, 6))
    
    #Original Singular Values
    plt.subplot(1, 2, 1)
    plt.plot(S, 'o-', label='Original Singular Values')
    plt.title('Original Singular Values')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    
    #Singular Values after Dimensionality Reduction
    plt.subplot(1, 2, 2)
    plt.plot(range(k), S_reduced, 'o-', color='orange', label='Retained Singular Values')
    plt.title('Retained Singular Values after Dimensionality Reduction')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.tight_layout()
    plt.legend()
    plt.show()
    
    return U, S, Vt, S_reduced




