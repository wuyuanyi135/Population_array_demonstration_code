import numpy as np
from numba import njit
from numba import typed, types
import numpy_indexed as npi
from scipy.stats import binned_statistic
from utils import volume_average_size

def compress_scipy_1d(n, interval, ws, kshape, log=False):
    sz = n[:, 0]
    if log:
        sz = np.log10(sz)
    dims_min = sz.min()
    dims_max = sz.max()
    cnts = (dims_max - dims_min) // interval + 1

    stat, edges, assignments = binned_statistic(sz, n[:, -1], 'sum', cnts)

    # use the index to partition
    partitions = npi.group_by(assignments).split(n)
    rows = []
    for p in partitions:
        if p.size == 0:
            continue
        rows.append(volume_average_size(p, ws, kshape))
    equivalent_rows = np.vstack(rows)

    return equivalent_rows

def compress(n, interval, ws, kshape, log=False):
    sz = n[:, :-1]
    if log:
        sz = np.log10(sz)
    dims_min = sz.min(axis=0)
    dims_max = sz.max(axis=0)
    cnts = (dims_max - dims_min) // interval + 1
    cumprod_dim = np.cumprod(np.hstack((np.array([1]), cnts)))[:-1]

    assignments = ((sz - dims_min) // interval * cumprod_dim).sum(axis=1)

    # use the index to partition
    partitions = npi.group_by(assignments).split(n)
    rows = []
    for p in partitions:
        if p.size == 0:
            continue
        rows.append(volume_average_size(p, ws, kshape))
    equivalent_rows = np.vstack(rows)

    return equivalent_rows


@njit()
def compress_jit(n, intervals, kshape, ws, compress_dim=0):
    ndim = n.shape[1] - 1
    mins = np.zeros((ndim,))
    maxs = np.zeros((ndim,))

    for i in range(ndim):
        l = n[:, i]
        mins[i] = l.max()
        maxs[i] = l.max()

    idx_prod_coefs = np.cumprod(maxs)
    idx_prod_coefs = np.hstack((np.array(1), idx_prod_coefs[0: ]))

    cnt_dict = typed.Dict.empty(
        key_type=types.int32,
        value_type=types.float64,
    )()
    vol_dict = typed.Dict.empty(
        key_type=types.int32,
        value_type=types.float64,
    )()
    non_adjust_dim_dict = typed.Dict.empty(
        key_type=types.int32,
        value_type=types.float64[:],
    )()

    nrow = n.shape[0]
    for ir in range(nrow):
        r = n[ir, :]
        sz = r[:-1]

        idx = (sz - mins) // intervals
        idx_flat = (idx_prod_coefs * idx).sum()

        cnt_dict[idx_flat] = r[-1]
        vol_dict[idx_flat] = r[-1] * (sz ** ws).prod() * kshape

        non_adj_sz = np.delete(sz, compress_dim)
        non_adjust_dim_dict[idx_flat] += r[-1] * non_adj_sz


    for i in range(len(cnt_dict)):
        pass
