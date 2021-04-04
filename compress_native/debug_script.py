import numpy as np
import sys
import os

sys.path.append(os.path.abspath("./cmake-build-debug"))
from compress_binding import compress as compress_native
# # test case 1
# N = np.array([
#     [1.0, 100],
#     [1.5, 200],
#     [2.5, 200],
# ])
# ws = np.array([3.])

# # test case 2 1d compression
# N = np.random.random((1000, 2))
# N[:, 0] *= 600
# N[:, 1] *= 1e6
# ws = np.array([3.])

# test case 3 2d compression
# np.random.seed(0)
N = np.random.random((1000, 3))
N[:, :-1] *= 10
N[:, -1] *= 1e6
ws = np.array([2., 1.])

interval = 1.0
kshape = 1.0
log = False

N_compress_native = compress_native(N, interval, ws, kshape, log)
print(N_compress_native)

count_after = N_compress_native[:, -1].sum()
count_before = N[:, -1].sum()
assert np.isclose(count_after, count_before), f"before={count_before}; after={count_after}"

vol_after = (N_compress_native[:, -1] * (N_compress_native[:, :-1]**ws).prod(axis=1)).sum()
vol_before = (N[:, -1] * (N[:, :-1]**ws).prod(axis=1)).sum()
assert np.isclose(vol_after, vol_before), f"before={vol_before}; after={vol_after}"