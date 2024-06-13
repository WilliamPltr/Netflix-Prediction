import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import time
from Module_Sparse import SparseMatrixTUPLES

def PlotTuple():
    sizes = [50, 200, 500, 1000, 1500]  
    times_add = [] 
    times_multiply = []  
    times_transpose = []  
    

    for size in sizes:
        #Generate random binary matrices for this size
        matrix_data = np.random.randint(2, size=(size, size))
        matrix_data2 = np.random.randint(2, size=(size, size))

        #Convert them to your SparseMatrixTUPLE representation
        A_sparse = SparseMatrixTUPLES(matrix_data)
        B_sparse = SparseMatrixTUPLES(matrix_data2)

        #Measure time for addition
        start = time.time()
        _ = A_sparse.add(B_sparse)
        times_add.append(time.time() - start)

        #Measure time for multiplication (use a vector for simplicity)
        vector = np.random.randint(2, size=(size,))
        start = time.time()
        _ = A_sparse.matrix_x_vector(vector)
        times_multiply.append(time.time() - start)

        #Measure time for transpose
        start = time.time()
        _ = A_sparse.transpose()
        times_transpose.append(time.time() - start)

    #Plotting results
    plt.plot(sizes, times_add, label='Addition', marker='o')
    plt.plot(sizes, times_multiply, label='Multiplication', marker='o')
    plt.plot(sizes, times_transpose, label='Transpose', marker='o')
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time for Sparse Matrix Operations (Tuple)')
    plt.ylim(0, 2)
    plt.xticks(sizes, labels=[str(size) for size in sizes])  # Setting custom labels for x-axis
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    PlotTuple()
