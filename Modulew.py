import numpy as np
import scipy as sp
import math
from scipy import linalg
from scipy.linalg import eigvals
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve




class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix
        self.rows = len(matrix)
        self.cols = len(matrix[0])
       
    def add(self, other):
        return Matrix(self.matrix + other.matrix)

    def sub(self, other):
        return Matrix(self.matrix - other.matrix)

    def scalar_multiply(self, scalar):
        return Matrix(self.matrix * scalar)

    def multiply(self, other):
        return Matrix(np.dot(self.matrix, other.matrix))

    def transpose(self):
        return Matrix(self.matrix.T)
    
    def matrix_x_vector(self,vector):
        #Check if multiplication can be done
        if self.cols != len(vector):
            return "Number of columns in the matrix must be equal to the length of the vector"
        result_vector = []
        for i in range(self.rows):
            dot_product = 0
            for j in range(self.cols): #Calculate dot product for the row i
                dot_product += int(self.matrix[i][j] * vector[j])
            result_vector.append(dot_product)
        return result_vector
    
    def matrix_x_matrix(self,other):
        #Check if multiplication is possible
        if self.cols != other.rows:
            return "Number of columns in the first matrix must be equal to the number of rows in the second matrix"
        result_matrix = []
        for i in range(self.rows):
            row_result = []
            for j in range(other.cols):
                dot_product = 0
                for k in range(self.cols):
                    #Calculate dot product for each cell
                    dot_product += self.matrix[i][k] * other.matrix[k][j]
                row_result.append(dot_product)
            result_matrix.append(row_result)
        return Matrix(result_matrix) #Return the result as a new Matrix object
    
    def norm_L1(self):
        #Calculate the sum of absolute values in each column
        sum_by_column = [0] * self.cols  #Initialize a list to store the sum of each column
        for j in range(self.cols):  #Loop over each column
            for i in range(self.rows):  #Loop over each row in the column
                sum_by_column[j] += abs(self.matrix[i][j])  #Add the absolute value of the element to the column's sum
        return max(sum_by_column)  #Return the highest sum among all columns

    def norm_L2(self):
        #Calculate the sum of squares of all elements in the matrix
        sum_of_squares = 0
        for i in range(self.rows):
            for j in range(self.cols):
                sum_of_squares += self.matrix[i][j] ** 2
        #Calculate the square root of the sum of squares to get the L2 norm
        return math.sqrt(sum_of_squares)

    def norm_Linf(self):
        #Calculate the sum of absolute values of each row
        sum_by_row = [0] * self.rows  #Initialize a list to store the sum of the rows
        for i in range(self.rows):
            for j in range(self.cols):
                sum_by_row[i] += abs(self.matrix[i][j])  #Add the absolute value of the element to the row's sum
        return max(sum_by_row)  #Return the maximum of the row sums

        #1.3.a
    def is_square(self):
        #Check if it's a square matrix
        return self.matrix.shape[0] == self.matrix.shape[1]
    def eigenvalues(self):
        #Square check for computation
        if self.is_square():
            return eigvals(self.matrix)
        else:
            return "The matrix must be a square Matrix !"
        #1.3.b
    def svd_compute(self):
        #Compute the SVD of the matrix (U,Sigma,V)
        U, S, V = np.linalg.svd(self.matrix)
        return U, S, V
        #1.4.a
 
    def solve_dense(self,C):
        #Solve the linear system and return the solution vector
        try:
            x = np.linalg.solve(self.matrix,C)
            return x
        except np.linalg.LinAlgError:
            # Catching the case where the matrix is singular
            print("Choose a non-singular matrix !")


    #Part 2
    @classmethod
    def generate_random_matrix(cls,rows,cols):
        matrix = np.random.randint(11, size=(rows,cols)) #Create a random matrix
        return cls(matrix)
    
    def transform_to_binary(self,threshold=5):
        #Function to apply the threshold
        binary = lambda x: 1 if x > threshold else 0
        binary_matrix = []
        for row in self.matrix:
            binary_row = []
            for i in row:
                binary_elem = binary(i) #Apply the thresholding function
                binary_row.append(binary_elem) #Add the result to the binary row
            binary_matrix.append(binary_row)
        return np.array(binary_matrix)

    #2.3 
    #After several tests, the fastest between Numpy and Scipy for SVD is Scipy, 
    #which is almost 100 times faster when iterated once, 
    #and is equal in execution time when iterated more than once.

    #Perform the SVD of our binary matrix
    #Transform BinaryMatrix to an instance
    def singular_values(self):
        BinaryMatrix=self.transform_to_binary()
        Bi = Matrix(BinaryMatrix)
        U, S, V = Bi.svd_compute()
        return S
    
    #2.4
    #Set the number of Singular Values needed as K
    def reduce_matrix(self):
        BinaryMatrix=self.transform_to_binary()
        Bi = Matrix(BinaryMatrix)
        k = 2 + 3*int(np.log((len(BinaryMatrix[0]))))
        U, S, V = Bi.svd_compute()
        U_reduced = U[:, :k]
        S_reduced = S[:k]
        Vt_reduced = V[:k, :]
        return U_reduced,S_reduced,Vt_reduced

    
    #2.5
    @classmethod
    def recommend(cls, liked_movie_index, VT, selected_movies_num):
        recommended = []

        # Calculate the dot product of the liked movie with all other movies
        for i in range(VT.shape[1]):  # Itérer sur les colonnes/films
            if i != liked_movie_index:
                dot_product_value = np.dot(VT[:, i], VT[:, liked_movie_index])
                recommended.append((i, dot_product_value))

        # Sort the movies based on the dot product values in descending order
        recommended.sort(key=lambda x: x[1], reverse=True)

        # Select the top recommended movies indices
        recommended_movies_indices = [rec[0] for rec in recommended[:selected_movies_num]]

        return recommended_movies_indices

        
    
class DenseMatrix(Matrix):
    def __init__(self,matrix):
        super.__init__(self,matrix)




