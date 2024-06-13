from Modulew import Matrix
import numpy as np
import scipy as sp
import math
from scipy import linalg
from scipy.linalg import eigvals
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

class SparseMatrixCSR(Matrix):
    def __init__(self,csr):
        #Initialize with a zero matrix matching the shape of csr
        super().__init__([[0 for _ in range(csr.shape[1])] for _ in range(csr.shape[0])])
        self.data = csr.data #Store non-zero entries
        self.indices = csr.indices #Column indices of non-zero elements
        self.indptr = csr.indptr #Index pointers to rows
        self.shape = csr.shape #Dimensions of the matrix

    def to_dense(self):
    #Convert a CSR SparseMatrix to a dense matrix (list of lists)
        dense_matrix = []
        #Initialize each row of the dense matrix
        for i in range(self.shape[0]):
            #Create a row of zeros
            row = [0] * self.shape[1]
            #Fill the row with non-zero values from CSR data
            for j in range(self.indptr[i], self.indptr[i+1]):
                col = self.indices[j]
                value = self.data[j]
                row[col] = value
            #Add the filled row to the dense matrix
            dense_matrix.append(row)
        return dense_matrix
    
    def add(self,other):
        if self.shape != other.shape:
            raise ValueError("Dimension problem.")

        #Initialize data structures for the resulting sparse matrix
        data_result = []
        indices_result = []
        indptr_result = [0]

        #Merge and add rows from both matrices
        for row in range(self.shape[0]):
            #Pointers to current element in each matrix's row
            i, j = self.indptr[row], other.indptr[row]
            #End indices for current row in each matrix
            i_1, j_1 = self.indptr[row + 1], other.indptr[row+1]

            while i < i_1 or j < j_1:
                #Case where we are only in the first matrix
                if j >= j_1 or (i < i_1 and self.indices[i] < other.indices[j]):
                    #Add element from first matrix
                    indices_result.append(self.indices[i])
                    data_result.append(self.data[i])
                    i += 1
                #Case where we are only in the second matrix
                elif i >= i_1 or (j < j_1 and other.indices[j] < self.indices[i]):
                    #Add element from second matrix
                    indices_result.append(other.indices[j])
                    data_result.append(other.data[j])
                    j += 1
                #Case where indices are in both matrices
                else:
                    #Elements are present in both, add their values
                    indices_result.append(self.indices[i])
                    data_result.append(self.data[i] + other.data[j])
                    i += 1
                    j+= 1
            #Update row pointer for the result
            indptr_result.append(len(data_result))
        result_csr = csr_matrix((data_result,indices_result,indptr_result),shape=self.shape)
        return result_csr
    
    def sub(self,other):
        if self.shape != other.shape:
            raise ValueError("Dimension mismatch.")
        #Initialize data structures for the resulting sparse matrix
        data_result = []
        indices_result = []
        indptr_result = [0]
        #Merge and add rows from both matrices
        for row in range(self.shape[0]):
            i, j = self.indptr[row], other.indptr[row] #Pointers to current element in each matrix's row
            i_end, j_end = self.indptr[row + 1], other.indptr[row+1] #End indices for current row in each matrix

            while i < i_end or j < j_end:
                #Case where we are only in the first matrix
                if j >= j_end or (i < i_end and self.indices[i] < other.indices[j]):
                    #Add element from first matrix
                    indices_result.append(self.indices[i])
                    data_result.append(self.data[i])  # Pas de changement de valeur
                    i += 1
                #Case where we are only in the second matrix
                elif i >= i_end or (j < j_end and other.indices[j] < self.indices[i]):
                    #Add element from second matrix
                    indices_result.append(other.indices[j])
                    data_result.append(-other.data[j])  # Changez la valeur en son négatif
                    j+= 1
                #Case where indices are in both matrices
                else:
                    #Elements are present in both, add their values
                    indices_result.append(self.indices[i])
                    data_result.append(self.data[i] - other.data[j])  # Soustraction des valeurs
                    i += 1
                    j += 1
            #Update row pointer for the result
            indptr_result.append(len(data_result))
        result_csr = csr_matrix((data_result,indices_result,indptr_result),shape=self.shape)
        return result_csr
    
    def matrix_x_vector(self,vector):
        if len(vector) != self.shape[1]:
            raise ValueError("Vector lenght must be qual to the number of column in matrix")
        
        result_vector = [0] * self.shape[0]  #Initialize the result vector with zeros

        for i in range(self.shape[0]): 
            for j in range(self.indptr[i], self.indptr[i+1]): #Iterate through the non-zero elements of the row
                result_vector[i] += float(self.data[j] * vector[self.indices[j]])
                
        return result_vector
    
    def matrix_x_matrix(self,other):
        if self.shape[1] != other.shape[0]:
            raise ValueError("Number of column of Matrix 1 has to match number of lines of Matrix 2")

        #Initialize structures for the resulting matrix
        data_result = []
        indices_result = []
        indptr_result = [0]

        for i in range(self.shape[0]):  #For each row in A
            row_values_dic = {}
            for j in range(self.indptr[i], self.indptr[i+1]):  #For each non-zero element in the row of A
                a_col = self.indices[j]  #Corresponding column in A
                a_val = self.data[j]  #Value of the element in A
                for k in range(other.indptr[a_col], other.indptr[a_col+1]):  #For each non-zero element in the column of B
                    b_row = other.indices[k]  #Corresponding row in B, which equals the column in A
                    b_val = other.data[k]  #Value of the element in B
                    if b_row not in row_values_dic:
                        row_values_dic[b_row] = 0
                    row_values_dic[b_row] += a_val * b_val
            
            #Add the calculated elements to the CSR data structure for C
            for col, val in sorted(row_values_dic.items()):
                indices_result.append(col)
                data_result.append(val)
            indptr_result.append(len(data_result)) #Update row pointer after each row is processed
        result_csr = csr_matrix((data_result,indices_result,indptr_result),shape=(self.shape[0],other.shape[1]))
        return result_csr

    def transpose(self):
        #The number of columns in the original matrix becomes the number of rows in the transposed
        num_rows = self.shape[1]
        #Initialize structures for the transposed matrix
        data_transposed = []
        indices_transposed = []
        indptr_transposed = [0] * (num_rows+1)

        #Count the number of elements in each "new row" (formerly columns)
        for i in self.indices:
            indptr_transposed[i+1] += 1

        #Accumulate values to get the starting indices of each new row
        for j in range(1, len(indptr_transposed)):
            indptr_transposed[j] += indptr_transposed[j-1]

        #Current position in each new row, used to insert values
        current_pos = indptr_transposed[:]
        
        for row in range(self.shape[0]):
            for j in range(self.indptr[row], self.indptr[row+1]):
                col = self.indices[j]
                val = self.data[j]
                
                #Find the position to insert the element in the transposed matrix
                pos = current_pos[col]
                data_transposed.append(val)
                indices_transposed.append(row)  #The old column is now a row
                current_pos[col] += 1
        result_csr = csr_matrix((data_transposed,indices_transposed,indptr_transposed),shape=(num_rows,self.shape[0]))
        return result_csr

class SparseMatrixTUPLES(Matrix):
    def __init__(self,matrix):
        Matrix.__init__(self,matrix)

    def to_sparse_representation(self):
        sparse_representation = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.matrix[i][j] != 0: #Check if the current element is not zero
                    sparse_representation.append((i,j,self.matrix[i][j])) #If not zero, add it as a tuple (row, column, value) to the list
        return sparse_representation
    
    @staticmethod
    def to_dense(sparse_representation,rows,cols):
        #Initialize an empty matrix
        dense_matrix = []
        #Fill the matrix with zeros, row by row
        for _ in range(rows):
            row = [] 
            for _ in range(cols):
                row.append(0) 
            dense_matrix.append(row)

        #Place the non-zero values in their respective positions
        for i,j,val in sparse_representation:
            dense_matrix[i][j] = val

        return SparseMatrixTUPLES(dense_matrix)
    
    @staticmethod
    def add_sub_sparse(sparse1,sparse2,operation="add"):
        #Regroup the common work for add or sub for optimization
        result = {}
        #Add all elements from sparse1 to result
        for i, j, val in sparse1:
            result[(i,j)] = val

        #Add or subtract elements from sparse2
        for i, j, val in sparse2:
            if (i,j) in result:
                if operation == "add":
                    result[(i,j)] += val
                elif operation == "sub":
                    result[(i,j)] -= val
            else:
                if operation == "sub":
                    result[(i,j)] = -val
                else:
                    result[(i,j)] = val

        #Filter zero elements after operation
        result_filtered = []  
        for (i, j), val in result.items():  # Loop through each item in the result dictionary
            if val != 0:  # Check if the value is non-zero
                result_filtered.append((i, j, val))  # Add the non-zero value tuple to the list
        return result_filtered

    def add(self,other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Dimension Problem")
        sparse_self = self.to_sparse_representation() #Change classic matrix in SparseMatrix
        sparse_other = other.to_sparse_representation() #Change classic matrix in SparseMatrix
        return SparseMatrixTUPLES.add_sub_sparse(sparse_self, sparse_other,operation="add")

    def sub(self,other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Dimension Problem")
        sparse_self = self.to_sparse_representation() #Change classic matrix in SparseMatrix
        sparse_other = other.to_sparse_representation() #Change classic matrix in SparseMatrix
        return SparseMatrixTUPLES.add_sub_sparse(sparse_self, sparse_other,operation="sub")

    def matrix_x_vector(self,vector):
        if len(vector) != self.cols:
            raise ValueError("Vecotr must have the same lenght as the number of column in matrix")
        #Convert the matrix to its sparse representation
        sparse_representation = self.to_sparse_representation()
        #Initialize the result vector with zeros
        result_vector = [0] * self.rows

        #Calculate the product
        for i,j,val in sparse_representation:
            result_vector[i] += val * vector[j]

        return result_vector
    
    def matrix_x_matrix(self,other):
        if self.cols != other.rows:
            raise ValueError("Number of columns of Matrix 1 muust be equal to number of lines of Matrix 2")

        sparse1 = self.to_sparse_representation()
        sparse2 = other.to_sparse_representation()

        result_dic = {}

        for i,j,val in sparse1:
            for k,l,val2 in sparse2:
                if j == k:  #Check if the column index of the first matrix matches the row index of the second matrix for valid multiplication
                    if (i,l) not in result_dic:
                        result_dic[(i,l)] = 0 #Initialize the result at position (i, l) if it doesn't exist
                    result_dic[(i,l)] += val * val2 #Multiply the values and add to the position in the result dictionary

        result_list = []
        for (i,j), val in result_dic.items():
            if val != 0:
                result_list.append((i,j,val)) #Filter zero values to maintain sparsity

        return result_list

    def transpose(self):
        #Create the list of tuples for the transpose
        transposed = []
        for i, j, val in self.to_sparse_representation(): #Loop on each non-zero element
            transposed.append((j,i,val)) #Inverse row and column
        return transposed
