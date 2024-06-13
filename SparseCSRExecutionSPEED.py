import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import time
from Module_Sparse import SparseMatrixCSR

def PlotCSR():
    sizes = [50, 200, 500, 1000, 1500]
    times_add = []
    times_multiply = []
    times_transpose = []

    for size in sizes:
        # Création de deux matrices creuses aléatoires
        A = sparse.random(size, size, density=0.03, format='csr')
        B = sparse.random(size, size, density=0.03, format='csr')
        C = np.random.randint(2, size=(size, 1))

        A_sparse = SparseMatrixCSR(A)
        B_sparse = SparseMatrixCSR(B)
        

        # Mesure du temps d'exécution pour l'addition
        start_time = time.time()
        _ = A_sparse.add(B_sparse)
        times_add.append(time.time() - start_time)

        # Mesure du temps d'exécution pour la multiplication
        start_time = time.time()
        _ = A_sparse.matrix_x_vector(C)
        times_multiply.append(time.time() - start_time)

        # Mesure du temps d'exécution pour la transposition
        start_time = time.time()
        _ = A_sparse.transpose()
        times_transpose.append(time.time() - start_time)

    plt.figure(figsize=(10, 8))
    plt.plot(sizes, times_add, label='Addition', marker='o')
    plt.plot(sizes, times_multiply, label='Vector multiplication', marker='o')
    plt.plot(sizes, times_transpose, label='Transposition', marker='o')

    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time for Sparse Matrix Operations (CSR)')
    plt.ylim(0, 2)
    plt.legend()
    plt.grid(True)
    plt.xticks(sizes, labels=[str(size) for size in sizes])  # Setting custom labels for x-axis
    plt.show()

if __name__ == "__main__":
    PlotCSR()