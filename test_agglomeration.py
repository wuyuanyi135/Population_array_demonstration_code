import numpy as np
from solvers import agglomeration_volume_grid_1d

L_edges = np.linspace(0, 100, 101)
L_centers = 0.5*(L_edges[:-1] + L_edges[1:])

n = np.zeros_like(L_centers)
n[:10] = 1

B, D = agglomeration_volume_grid_1d(n, L_edges, L_centers, 1.0)

print (B, D)