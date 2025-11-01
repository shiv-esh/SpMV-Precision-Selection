import numpy as np
from scipy import sparse
from scipy.io import mmwrite
import random

# Ask the user for the number of matrices to generate
n_matrices = int(input("Enter the number of matrices to generate: "))

# Generate and save n different matrices with random sizes and density levels
for i in range(n_matrices):
    n_rows = random.randint(500, 2000)
    n_cols = random.randint(500, 2000)
    density = random.uniform(0.05, 0.4)
    random_matrix = sparse.random(n_rows, n_cols, density=density, format='coo', dtype=np.float64)
    filename = f'random_mtx_nc{i + 1}.mtx'
    mmwrite(filename, random_matrix)
    print(f"Sparse matrix with size ({n_rows}, {n_cols}) and density {density:.2f} saved to '{filename}'")