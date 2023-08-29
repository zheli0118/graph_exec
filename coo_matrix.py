import numpy as np
from scipy.sparse import coo_matrix
data = np.ones(3)
row = np.array([0, 1, 2])
col = np.array([0, 1, 2])

A = coo_matrix((data, (row, col)), shape=(3, 3))

print(A.toarray())