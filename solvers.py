import numpy as np
from numba import njit
from enum import Enum


# 1D
@njit()
def van_leer_limiter(theta):
    abs_theta = np.abs(theta)
    return (abs_theta + theta) / (1 + abs_theta)


@njit()
def theta(n):
    t = (n[1:-1] - n[0:-2]) / (n[2:] - n[1:-1])
    t[np.isnan(t)] = 2
    t[np.isinf(t)] = 0
    return t


@njit()
def hr_grid_forward_step_size_independent_1d(n, dL, dt, g):
    if g == 0:
        return n

    dissolve = False
    if g < 0:
        n = n[::-1]
        g = -g
        dissolve = True

    # padding the boundary conditions
    n_pad = np.zeros((n.shape[0] + 3))
    n_pad[2:-1] = n

    coef = dt / dL * g
    theta_n = theta(n_pad[1:])
    limiter_n = van_leer_limiter(theta_n)
    theta_n_1 = theta(n_pad[0:-1])
    limiter_n_1 = van_leer_limiter(theta_n_1)
    n = n - coef * (n_pad[2:-1] - n_pad[1:-2]) - coef / 2 * (1 - coef) * (
            (n_pad[3:] - n_pad[2:-1]) * limiter_n - (n_pad[2:-1] - n_pad[1:-2]) * limiter_n_1)

    if dissolve:
        return n[::-1]
    else:
        return n


@njit()
def hr_grid_forward_step_size_dependent_1d(n, dL, dt, G):
    if np.all(G == 0):
        return n

    dissolve = False
    if np.any(G < 0):
        n = n[::-1]
        G = -G
        dissolve = True

    # padding the boundary conditions
    n_pad = np.zeros((n.shape[0] + 3))
    n_pad[2:-1] = n

    G_pad = np.empty((G.shape[0] + 3))
    G_pad[2:-1] = G
    G_pad[:2] = G[0]
    G_pad[-1] = G[-1]

    GN_pad = G_pad * n_pad

    coef = dt / dL
    theta_n = theta(n_pad[1:])
    limiter_n = van_leer_limiter(theta_n)
    theta_n_1 = theta(n_pad[0:-1])
    limiter_n_1 = van_leer_limiter(theta_n_1)
    n = n - coef * (GN_pad[2:-1] - GN_pad[1:-2]) - coef / 2 * (1 - coef) * (
            (GN_pad[3:] - GN_pad[2:-1]) * limiter_n - (GN_pad[2:-1] - GN_pad[1:-2]) * limiter_n_1)

    if dissolve:
        return n[::-1]
    else:
        return n


@njit()
def hr_grid_forward_step_size_independent_2d(n, dL, dt, gs):
    for i in range(n.shape[1]):
        # along dim0 (rows) for each col
        n[:, i] = hr_grid_forward_step_size_independent_1d(n[:, i], dL, dt, gs[0])

    for i in range(n.shape[0]):
        # along dim1 (cols) for each row
        n[i, :] = hr_grid_forward_step_size_independent_1d(n[i, :], dL, dt, gs[1])

    return n


@njit()
def uw_grid_forward_step_size_independent_1d(n, dL, dt, g):
    if g == 0:
        return n

    dissolve = False
    if g < 0:
        n = n[::-1]
        g = -g
        dissolve = True

    # padding the boundary conditions
    n_pad = np.zeros((n.shape[0] + 1))
    n_pad[1:] = n

    coef = dt / dL * g
    n = n - coef * (n_pad[1:] - n_pad[:-1])

    if dissolve:
        return n[::-1]
    else:
        return n


@njit()
def uw_grid_forward_step_size_dependent_1d(n, dL, dt, G):
    if np.all(G == 0):
        return n

    dissolve = False
    if np.any(G < 0):
        n = n[::-1]
        G = -G
        dissolve = True

    # padding the boundary conditions
    n_pad = np.zeros((n.shape[0] + 1))
    n_pad[1:] = n

    G_pad = np.empty((G.shape[0] + 1))
    G_pad[1:] = G
    G_pad[0] = G[0]
    GN_pad = G_pad * n_pad

    coef = dt / dL
    n = n - coef * (GN_pad[1:] - GN_pad[:-1])

    if dissolve:
        return n[::-1]
    else:
        return n


@njit()
def uw_grid_forward_step_size_independent_2d(n, dL, dt, gs):
    for i in range(n.shape[1]):
        # along dim0 (rows) for each col
        n[:, i] = uw_grid_forward_step_size_independent_1d(n[:, i], dL, dt, gs[0])

    for i in range(n.shape[0]):
        # along dim1 (cols) for each row
        n[i, :] = uw_grid_forward_step_size_independent_1d(n[i, :], dL, dt, gs[1])

    return n


# population array method
# @njit()
def pa_forward_step(N, dt, gs):
    N[:, :-1] += gs * dt
    if np.any(gs < 0):
        idx = np.zeros((N.shape[0],), dtype=np.bool8)
        for i in range(N.shape[1] - 1):
            idx = np.logical_or(idx, N[:, i] <= 0)
        N = N[~idx]
    return N


###############
# Agglomeration
###############
class AgglomerationKernel(Enum):
    CONSTANT = 0


class GridType(Enum):
    VOLUME = 0
    SIZE = 1
    SQUARE = 2


@njit()
def constant_aggl_kernel(x_1, x_2, x_aggl, v_1, v_2, v_aggl):
    return 1


@njit()
def agglomeration_grid_1d(
        n: np.ndarray,
        edge_grids: np.ndarray,
        center_grids: np.ndarray,
        kernel_coef: float,
        aggl_dim_power: int = 1,
        kernel_type: AgglomerationKernel = AgglomerationKernel.CONSTANT,
        min_count=1e-9,
):
    """

    :param other_dims: non-agglomerating dimensions
    :param n:
    :param edge_grids:
    :param center_grids:
    :param kernel_coef:
    :param aggl_dim_power: how the edge grid should be treated along the agglomeration dimension
    :param kernel_type:
    :return:
    """
    if kernel_type == AgglomerationKernel.CONSTANT:
        kernel_fcn = constant_aggl_kernel
    else:
        raise ValueError("Unsupported agglomeration kernel type")

    N = np.diff(edge_grids) * n
    M = n.size  # equal to center grid number.
    D = np.zeros_like(n)
    B = np.zeros_like(n)
    # oop = np.prod(np.asarray(other_dims ** other_dims_powers)) # oop can be cancelled everywhere.
    # Calculate death rate
    for i, Ni in enumerate(N):
        xi = center_grids[i]
        vi = xi ** aggl_dim_power
        if Ni < min_count:
            continue

        for j, Nj in enumerate(N):
            if Nj < min_count:
                continue
            xj = center_grids[j]
            vj = xj ** aggl_dim_power

            v_aggl = vi + vj
            x_aggl = (v_aggl) ** (1 / aggl_dim_power)

            coef = kernel_fcn(xi, xj, x_aggl, vi, vj, v_aggl) * kernel_coef
            D[i] += coef * Ni * Nj

    # Birth is calculated by agglomerating the smaller crystals
    # store all combinations first
    agglomerate = np.empty((int(M * (M + 1) / 2), 3))  # first col: x_aggl, second: N
    idx = 0
    for k, Nk in enumerate(N):
        # skip the crystal smaller than k to prevent duplication (1+2 and 2+1 are the same)
        xk = center_grids[k]
        vk = xk ** aggl_dim_power
        if Nk < min_count:
            continue
        for j in range(k, M):
            Nj = N[j]
            if Nj < min_count:
                continue
            xj = center_grids[j]
            vj = xj ** aggl_dim_power

            v_aggl = vk + vj
            x_aggl = v_aggl ** (1 / aggl_dim_power)

            # find index of xi, the agglomerate
            coef = kernel_fcn(xj, xk, x_aggl, vj, vk, v_aggl) * kernel_coef
            val = coef * N[j] * N[k]

            if j == k:
                val /= 2
            agglomerate[idx, 0] = x_aggl
            agglomerate[idx, 1] = v_aggl
            agglomerate[idx, 2] = val
            idx += 1

    # assemble the agglomerate to the original grid
    x_aggl = agglomerate[:, 0]
    v_aggl = agglomerate[:, 1]
    val_aggl = agglomerate[:, 2]

    for i in range(M):
        filter_range_low = edge_grids[i]
        filter_range_high = edge_grids[i+1]
        in_range_idx = (x_aggl >= filter_range_low) & (x_aggl < filter_range_high)
        x_filtered = x_aggl[in_range_idx]
        v_filtered = v_aggl[in_range_idx]
        val_filtered = val_aggl[in_range_idx]

        vi = center_grids[i] ** aggl_dim_power
        phi = v_filtered / vi

        B[i] = (phi * val_filtered).sum()

    # here B and D is not compatible with the number density.
    gd_diff = np.diff(edge_grids)
    B_nd = B / gd_diff
    D_nd = D / gd_diff
    return B_nd, D_nd



@njit
def agglomeration_grid_2d(
        n: np.ndarray,
        edge_grids,
        center_grids,
        kernel_coefs: np.ndarray,
        aggl_dim_powers: np.ndarray,
        kernel_type: AgglomerationKernel = AgglomerationKernel.CONSTANT,
        min_count=1e-9,
):
    """

    :param n:
    :param edge_grids: tuple of each dimension
    :param center_grids: tuple of each dimension
    :param kernel_coefs: kernel coefficients of n dimensions
    :param kernel_type:
    :param grid_type:
    :return:
    """
    ndim0, ndim1 = n.shape
    B = np.zeros_like(n)
    D = np.zeros_like(n)

    for i in range(ndim0):
        other_edge = edge_grids[1]
        other_edge_diff = (other_edge[i + 1] - other_edge[i])

        n_1d = n[i, :] * other_edge_diff
        edge = edge_grids[0]
        center = center_grids[0]
        B_1d, D_1d = agglomeration_grid_1d(n_1d, edge, center, kernel_coefs[0], aggl_dim_powers[0],
                                           kernel_type=kernel_type,
                                           min_count=min_count)

        B[i, :] += B_1d / other_edge_diff
        D[i, :] += D_1d / other_edge_diff

    for i in range(ndim1):
        other_edge = edge_grids[0]
        other_edge_diff = (other_edge[i + 1] - other_edge[i])

        n_1d = n[:, i] * other_edge_diff
        edge = edge_grids[1]
        center = center_grids[1]
        B_1d, D_1d = agglomeration_grid_1d(n_1d, edge, center, kernel_coefs[1], aggl_dim_powers[1],
                                           kernel_type=kernel_type, min_count=min_count)

        B[:, i] += B_1d / other_edge_diff
        D[:, i] += D_1d / other_edge_diff

    return B, D


@njit()
def agglomeration_pa_1d(
        N: np.ndarray,
        kernel_coef: float,
        dim_powers: np.ndarray = np.array([1]),
        aggl_dim: int = 0,
        kernel_type: AgglomerationKernel = AgglomerationKernel.CONSTANT,
        min_count=1e-9
):
    included_rows = N[:, -1] >= min_count
    N_original = N
    N = N[included_rows]

    if kernel_type == AgglomerationKernel.CONSTANT:
        kernel_fcn = constant_aggl_kernel
    else:
        raise ValueError("Unsupported agglomeration kernel type")

    nrows, ncols = N.shape

    D = np.zeros((nrows,))
    B = np.zeros((nrows * (nrows + 1) // 2, ncols))
    idx = 0
    for i, row in enumerate(N):
        xi = row[aggl_dim]
        vi = (row[:-1] ** dim_powers).prod()
        Ni = row[-1]
        for j in range(i, nrows):
            xj = N[j, aggl_dim]
            Nj = N[j, -1]
            vj = (N[j, :-1] ** dim_powers).prod()

            v_aggl = vi + vj
            # this is for computing the mean size of the non agglomerating dimensions
            x_mean = 0.5 * (row[:-1] + N[j, :-1])
            x_mean[aggl_dim] = 1.  # exclude the agglomeration dimension from the volume calculation

            x_aggl = (v_aggl / (x_mean ** dim_powers).prod()) ** (1 / dim_powers[aggl_dim])
            x_mean[aggl_dim] = x_aggl

            coef = kernel_fcn(xi, xj, x_aggl, vi, vj, v_aggl) * kernel_coef
            consumption_rate = coef * Ni * Nj

            if consumption_rate < min_count:
                continue

            D[i] += consumption_rate
            D[j] += consumption_rate

            B[idx, :-1] = x_mean
            B[idx, -1] = consumption_rate
            idx += 1
    B = B[B[:, 0] != 0]
    D_original = np.zeros((N_original.shape[0],))
    D_original[included_rows] = D
    return B, D_original


@njit()
def agglomeration_pa_2d(
        N,
        edge_grids,
        kernel_coefs: np.ndarray,
        aggl_dim_powers: np.ndarray,
        kernel_type: AgglomerationKernel = AgglomerationKernel.CONSTANT,
        min_count=1e-9
):
    nrows = N.shape[0]

    D_total = np.zeros((nrows,))
    B = []

    # agglomerating along dim0
    # the other dim is used to filter the crystals with similar sizes of non agglomerating dimension.
    e = edge_grids[1]
    e_grid = np.arange(0, N[:, 1].max(), e)
    n_edge_nodes = e_grid.size
    for i in range(n_edge_nodes - 1):
        high_lim = e_grid[i + 1]
        low_lim = e_grid[i]
        filtered_idx = (N[:, 1] <= high_lim) & (N[:, 1] > low_lim)
        N_filtered = N[filtered_idx]
        B_, D = agglomeration_pa_1d(N_filtered, kernel_coefs[0], aggl_dim_powers, aggl_dim=0, kernel_type=kernel_type,
                                    min_count=min_count)
        D_total[filtered_idx] += D
        if len(B_) > 0:
            B.append(B_)

    e = edge_grids[0]
    e_grid = np.arange(0, N[:, 1].max(), e)
    n_edge_nodes = e_grid.size
    for i in range(n_edge_nodes - 1):
        high_lim = e_grid[i + 1]
        low_lim = e_grid[i]
        filtered_idx = (N[:, 0] <= high_lim) & (N[:, 0] > low_lim)
        N_filtered = N[filtered_idx]
        B_, D = agglomeration_pa_1d(N_filtered, kernel_coefs[1], aggl_dim_powers, aggl_dim=1, kernel_type=kernel_type,
                                    min_count=min_count)
        D_total[filtered_idx] += D
        if len(B_) > 0:
            B.append(B_)

    return B, D_total


###############
# Breakage
###############
class BreakageKernel(Enum):
    LINEAR = 0
    SQUARE = 1


class BreakageFrequencyDistribution(Enum):
    UNIFORM = 0


@njit
def linear_brk_kernel(xp):
    return xp


@njit
def square_brk_kernel(xp):
    return xp ** 2


@njit
def brk_nik_uniform(edge_grids, center_grids, i, k):
    vk = center_grids[k]

    if i == 0:
        x_diff = center_grids[i + 1] - center_grids[i]
    elif i == center_grids.size - 1:
        x_diff = center_grids[i] - center_grids[i - 1]
    else:
        x_diff = center_grids[i + 1] - center_grids[i - 1]
    return x_diff / vk


@njit
def breakage_grid_1d(
        n: np.ndarray,
        edge_grids: np.ndarray,
        center_grids: np.ndarray,
        kernel_coef: float,
        kernel_type: BreakageKernel = BreakageKernel.SQUARE,
        break_dim_power: int = 1,
        other_dims: np.ndarray = np.array([0.]),
        other_dims_power: np.ndarray = np.array([0.]),
        frequency_dist: BreakageFrequencyDistribution = BreakageFrequencyDistribution.UNIFORM,
):
    if kernel_type == BreakageKernel.LINEAR:
        kernel_fcn = linear_brk_kernel
    elif kernel_type == BreakageKernel.SQUARE:
        kernel_fcn = square_brk_kernel
    else:
        raise ValueError("Unsupported breakage kernel type")

    if frequency_dist == BreakageFrequencyDistribution.UNIFORM:
        nik_fcn = brk_nik_uniform
    else:
        raise ValueError("Unsupported frequency distribution")
    N = np.diff(edge_grids) * n
    M = n.size  # equal to center grid number.
    D = np.zeros_like(n)
    B = np.zeros_like(n)

    # transform edge and center grids to volume
    oop = np.prod(np.asarray(other_dims ** other_dims_power))
    edge_grids_v = edge_grids ** break_dim_power * oop
    center_grids_v = center_grids ** break_dim_power * oop
    for i, Ni in enumerate(N):
        xi = center_grids[i]
        vi = (xi ** break_dim_power) * oop
        coef = kernel_fcn(vi) * kernel_coef
        D[i] = coef * Ni

        for k in range(i, M):
            xk = center_grids[k]
            vk = (xk ** break_dim_power) * oop
            Nk = N[k]
            nik = nik_fcn(edge_grids_v, center_grids_v, i, k)
            eta_k = kernel_fcn(vk) * kernel_coef
            B[i] += nik * eta_k * Nk

    # here B and D is not compatible with the number density.
    gd_diff = np.diff(edge_grids)
    B_nd = B / gd_diff
    D_nd = D / gd_diff
    return B_nd, D_nd


@njit
def breakage_grid_2d(
        n: np.ndarray,
        edge_grids: np.ndarray,
        center_grids: np.ndarray,
        kernel_coefs: np.ndarray,
        break_dim_powers: np.ndarray = np.array([2, 1]),
        kernel_type: BreakageKernel = BreakageKernel.SQUARE,
        frequency_dist: BreakageFrequencyDistribution = BreakageFrequencyDistribution.UNIFORM,
):
    ndim0, ndim1 = n.shape
    B = np.zeros_like(n)
    D = np.zeros_like(n)

    for i in range(ndim0):
        other_edge = edge_grids[1]
        other_edge_diff = (other_edge[i + 1] - other_edge[i])
        other_center = center_grids[1]
        n_1d = n[i, :] * other_edge_diff
        edge = edge_grids[0]
        center = center_grids[0]

        B_1d, D_1d = breakage_grid_1d(n_1d, edge, center, kernel_coefs[0], kernel_type, break_dim_powers[0],
                                      other_center[i], break_dim_powers[1], frequency_dist)

        B[i, :] += B_1d / other_edge_diff
        D[i, :] += D_1d / other_edge_diff

    for i in range(ndim1):
        other_edge = edge_grids[0]
        other_edge_diff = (other_edge[i + 1] - other_edge[i])
        other_center = center_grids[0]
        n_1d = n[:, i] * other_edge_diff
        edge = edge_grids[1]
        center = center_grids[1]
        B_1d, D_1d = breakage_grid_1d(n_1d, edge, center, kernel_coefs[1], kernel_type, break_dim_powers[1],
                                      other_center[i], break_dim_powers[0], frequency_dist)

        B[:, i] += B_1d / other_edge_diff
        D[:, i] += D_1d / other_edge_diff

    return B, D


class DaughterGridType(Enum):
    LINEAR = 0
    LOG = 1


@njit
def logspace(logstart, logend, n):
    grid = np.linspace(logstart, logend, int(n))
    return 10. ** grid


@njit
def log_mean(edges):
    log_edges = np.log10(edges)
    center = 0.5 * (log_edges[1:] + log_edges[:-1])
    return 10. ** center


@njit
def breakage_pa_1d(
        N: np.ndarray,
        kernel_coef: float,
        min_count=1e-9,
        dim_powers: np.ndarray = np.array([1.]),
        break_dim: int = 0,
        kernel_type: BreakageKernel = BreakageKernel.SQUARE,
        frequency_dist: BreakageFrequencyDistribution = BreakageFrequencyDistribution.UNIFORM,
        daughter_grid_type: DaughterGridType = DaughterGridType.LOG,
        daughter_grid_factor=10.,
        daughter_grid_min=1,
        daughter_grid_max=10000,
        daughter_grid_low_limit=1e-5,
):
    included_rows = (N[:, -1] >= min_count) & (N[:, break_dim] >= daughter_grid_low_limit * 2)  # VOLUME ONLY
    N_original = N
    N = N[included_rows]

    if kernel_type == BreakageKernel.SQUARE:
        kernel_fcn = square_brk_kernel
    elif kernel_type == BreakageKernel.LINEAR:
        kernel_fcn = linear_brk_kernel
    else:
        raise ValueError("Unsupported breakage kernel type")

    if daughter_grid_type == DaughterGridType.LOG:
        sz = np.log10(N[:, break_dim])
        min_sz = np.log10(daughter_grid_low_limit)
    elif daughter_grid_type == DaughterGridType.LINEAR:
        sz = N[:, break_dim]
        min_sz = daughter_grid_low_limit
    else:
        raise NotImplementedError("Unsupported daughter grid")

    daughter_counts = np.ceil((sz - min_sz) * daughter_grid_factor)
    daughter_counts[daughter_counts < daughter_grid_min] = daughter_grid_min
    daughter_counts[daughter_counts > daughter_grid_max] = daughter_grid_max

    nrows, ncols = N.shape
    D = np.zeros((nrows,))
    B = np.empty((int(daughter_counts.sum()), ncols))
    idx = 0

    for i, row in enumerate(N):
        # xi = row[break_dim]
        row_sz = row[:-1]
        vi = (row_sz ** dim_powers).prod()
        row_ = row_sz.copy()
        row_[break_dim] = daughter_grid_low_limit
        v_low_limit = (row_ ** dim_powers).prod()
        Ni = row[-1]
        rate = Ni * kernel_coef * kernel_fcn(vi)
        D[i] = rate

        n_daughter = int(daughter_counts[i])
        if daughter_grid_type == DaughterGridType.LOG:
            daughter_grid_edge = logspace(np.log10(v_low_limit), np.log10(vi), n_daughter + 1)
            daughter_grid_center = log_mean(daughter_grid_edge)
        elif daughter_grid_type == DaughterGridType.LINEAR:
            daughter_grid_edge = np.linspace(v_low_limit, vi, n_daughter + 1)
            daughter_grid_center = 0.5 * (daughter_grid_edge[1:] + daughter_grid_edge[:-1])
        else:
            raise NotImplementedError("Unsupported grid type")
        # convert from volume to original grid.
        partial_volume_dim = row_sz.copy()
        partial_volume_dim[break_dim] = 1.
        partial_volume = (partial_volume_dim ** dim_powers).prod()
        daughter_grid_center = (daughter_grid_center / partial_volume) ** (1 / dim_powers[break_dim])

        daughter_grid_all_internal_coors = np.empty((n_daughter, ncols - 1))
        daughter_grid_all_internal_coors[:] = row_sz
        daughter_grid_all_internal_coors[:, break_dim] = daughter_grid_center

        diff = daughter_grid_edge[1:] - daughter_grid_edge[:-1]

        birth_rates = rate * 2 * diff / vi
        B[idx:idx + n_daughter, :-1] = daughter_grid_all_internal_coors
        B[idx:idx + n_daughter, -1] = birth_rates
        idx += n_daughter

    D_original = np.zeros((N_original.shape[0],))
    D_original[included_rows] = D
    return B, D_original


@njit
def np_any_row_wise(N):
    ret = np.zeros((N.shape[0],), dtype=np.bool8)
    for i in range(N.shape[1]):
        ret = np.logical_or(ret, N[:, i])
    return ret


@njit
def breakage_pa_2d(
        N: np.ndarray,
        kernel_coefs: np.ndarray,
        min_count=1e-9,
        dim_powers: np.ndarray = np.array([2., 1.]),
        kernel_type: BreakageKernel = BreakageKernel.SQUARE,
        frequency_dist: BreakageFrequencyDistribution = BreakageFrequencyDistribution.UNIFORM,
        daughter_grid_type: DaughterGridType = DaughterGridType.LOG,
        daughter_grid_factor=10.,
        daughter_grid_min=1,
        daughter_grid_max=10000,
        daughter_grid_low_limit=1e-5,
):
    nrows, ncols = N.shape
    B = []
    D = np.zeros((nrows,))

    for idim in range(ncols - 1):
        B_, D_ = breakage_pa_1d(N, kernel_coefs[idim], min_count, dim_powers, idim, kernel_type, frequency_dist,
                                daughter_grid_type, daughter_grid_factor, daughter_grid_min, daughter_grid_max,
                                daughter_grid_low_limit)
        B.append(B_)
        D += D_
    return B, D
