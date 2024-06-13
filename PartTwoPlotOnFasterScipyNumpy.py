import numpy as np
from scipy.linalg import svd as scipy_svd
from numpy.linalg import svd as numpy_svd
import matplotlib.pyplot as plt
import time

def compare_svd_methods(users=100, movies=50):
    
    #Step 1: Generate a random binary matrix
    ratings = np.random.randint(2, size=(users, movies))
    
    #Step 2: Perform SVD using both scipy and numpy, measure computation time
    start_time = time.time()
    U_scipy, s_scipy, Vt_scipy = scipy_svd(ratings, full_matrices=False)
    scipy_time = time.time() - start_time
    
    start_time = time.time()
    U_numpy, s_numpy, Vt_numpy = numpy_svd(ratings, full_matrices=False)
    numpy_time = time.time() - start_time
    
    #Compare computation time and choose the faster method
    if scipy_time < numpy_time:
        faster_method = 'SciPy'
        s_values, U, s, Vt = s_scipy, U_scipy, s_scipy, Vt_scipy
    else:
        faster_method = 'NumPy'
        s_values, U, s, Vt = s_numpy, U_numpy, s_numpy, Vt_numpy
    
    print(f"\nSciPy SVD computation time: {scipy_time:.4f} seconds")
    print(f"NumPy SVD computation time: {numpy_time:.4f} seconds")
    print(f"Faster method: {faster_method}")

    print("\nThe singulars values of our matrix are :")
    print("\n",s_scipy)
    
    # Step 3: Plot singular values
    plt.figure(figsize=(10, 6))
    plt.plot(s_values, marker='o', linestyle='-', color='b')
    plt.title('Singular Values Distribution')
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value Magnitude')
    plt.grid(True)
    plt.show()
    
    return (U, s, Vt, scipy_time, numpy_time)

# Example usage
#U, s, Vt, scipy_time, numpy_time = compare_svd_methods(100, 50)
