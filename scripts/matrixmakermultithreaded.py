import numpy as np
from decimal import Decimal
from scipy.io import mmwrite
import struct
from scipy.sparse import coo_matrix
import sys


# This script generates a CSR matrix according to the characteristics given by the user.
# There are a few modes to generate a matrix
# 1. Fill up each row fully with non-zeros before proceeding to the next row
# 2. Fill up each column fully with non-zeros before proceeding to the next column
# 3. Fill up matrix with uniform distribution of non-zeros across rows and columns
# Comment out/in the modes according to which you prefer
# Usage: python3 matrixmakermultithreaded_py <sparsity> <numRows> <numCols>

def write_file(file_path, data_array, data_type='d'):
    with open(file_path, 'wb') as f:
        f.write(struct.pack(f'{len(data_array)}{data_type}', *data_array))


def generate_sparse_matrix(numRows, numCols, sparsity):
    num_nonzero_elements = int(numRows * numCols * (1 - sparsity))
    row_indices = []
    col_indices = []

    dict = set()
    P = 8500

    # 1. for packing rows
    rows = numRows // P
    for j in range(rows):
        for i in range(num_nonzero_elements // rows):
            row_indices.append(j * P)
            col_indices.append(i)
            dict.add((j, i))

    # 2. for packing cols
    # cols = numCols // P
    # for j in range(cols):
    #     for i in range(num_nonzero_elements // cols):
    #         col_indices.append(j * P)
    #         row_indices.append(i)
    #         dict.add((j, i))

    # 3. uniform distribution (for sparsity >= 50%)
    # for i in range(num_nonzero_elements):
    #     row, col = np.random.randint(numRows), np.random.randint(numCols)
    #     if (row, col) not in dict:
    #         dict.add((row, col))
    #         row_indices.append(row)
    #         col_indices.append(col)

    print(f"Expected Density = {1 - sparsity}. Achieved Density = {len(dict) / (numRows * numCols)}")
    print(f"NNZ = {len(dict)}")
    print(f"Achieved Density = {1 - sparsity}")

    values = np.ones(len(dict))
    sparse_matrix = coo_matrix((values, (row_indices, col_indices)), shape=(numRows, numCols))

    # uniform distribution (for sparsity < 50%)
    # sparse_matrix = random(numRows, numCols, density=1 - sparsity, format='coo')
    # print(f"Achieved Density = {1 - sparsity}")
    # print(f"NNZ = {(1 - sparsity) * numRows * numCols}")

    return sparse_matrix


if len(sys.argv) != 4:
    sparsity = 0.90
    numRows = 10000
    numCols = numRows
else:
    sparsity = Decimal(sys.argv[1])
    numRows = int(sys.argv[2])
    numCols = int(sys.argv[3])

print("Generating Sparse Matrix....")
sparse_matrix = generate_sparse_matrix(numRows, numCols, sparsity)

# Write the sparse matrix in Matrix Market format
mmwrite("sparse.mtx", sparse_matrix, precision=1)
