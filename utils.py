from numba import njit
import numpy as np


@njit()
def per_particle_volume(n, ws, kshape):
    """
    ignore the count, calculate the per-particle volume
    :param n:
    :return:
    """
    out = np.ones((n.shape[0],))
    for i in range(n.shape[1] - 1):
        out *= n[:, i] ** ws[i]
    return out * kshape


@njit()
def particle_volume(n, ws, kshape):
    return per_particle_volume(n, ws, kshape) * n[:, -1]


@njit()
def volume_fraction(n, ws, kshape):
    return particle_volume(n, ws, kshape).sum(axis=0)

@njit()
def volume_average_size(n, ws, kshape):
    # return 1d scalar of one row
    cnt = n[:, -1].sum()

    v = volume_fraction(n, ws, kshape)

    ncols = n.shape[1]

    ret = np.empty((ncols,))

    if ws.size == 1:
        # one dimensional
        ret[0] = (v / cnt / kshape) ** (1 / ws[0])
        ret[1] = cnt

        return ret
    else:
        # multi-dimensional N
        non_first_dim_mean_sizes = (n[:, 1:-1] * np.expand_dims(n[:, -1], 1)).sum(axis=0) / cnt
        particle_average_volume = v / cnt / kshape
        non_first_dim_prod = np.prod(non_first_dim_mean_sizes ** ws[1:])

        # exclude the effect of the non first dimensions. The modified particle_average_volume can be used to
        # calculate the first dimension by reciprocal power of the first dimension
        particle_average_volume = particle_average_volume / non_first_dim_prod

        first_dim_size = particle_average_volume ** (1 / ws[0])

        ret[0] = first_dim_size
        ret[1:-1] = non_first_dim_prod
        ret[-1] = cnt

        return ret
