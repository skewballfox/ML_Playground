import numpy as np

adjacency = np.array([[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]])
degree = np.diag([3, 2, 2, 1])
incidence = np.array([[-1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, -1, 0], [-1, 0, 0, 1]])

laplacian_from_incidence = incidence.T @ incidence
laplacian_by_definition = degree - adjacency
print(laplacian_from_incidence)
print(laplacian_by_definition)
print(np.linalg.eig(laplacian_by_definition))
