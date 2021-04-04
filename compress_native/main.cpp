#include <cmath>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::array compress(py::array_t<double> N, double interval, py::array_t<double> ws, double kshape, bool log = false) {
    py::buffer_info info = N.request();
    auto ptr = (double *) (info.ptr);
    auto n_rows = info.shape[0];
    auto n_col = info.shape[1];
    auto n_dim = n_col - 1;

    auto sz_array = std::vector<double>(n_dim * n_rows);

    py::buffer_info ws_info = ws.request();
    auto *ws_ptr = (double *) ws_info.ptr;

    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_dim; ++j) {
            auto idx_ptr = i * n_col + j;
            auto idx_dim = i * n_dim + j;

            sz_array[idx_dim] = log ? log10(ptr[idx_ptr]) : ptr[idx_ptr];
        }
    }

    // find the max and min of each dimension
    auto dim_min = std::vector<double>(n_dim);
    auto dim_max = std::vector<double>(n_dim);

    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_dim; ++j) {
            auto idx_dim = i * n_dim + j;
            auto val = sz_array[idx_dim];
            if (dim_min[j] > val) {
                dim_min[j] = val;
            }

            if (dim_max[j] < val) {
                dim_max[j] = val;
            }
        }
    }

    auto dim_cnts = std::vector<ssize_t>(n_dim);
    for (int i = 0; i < dim_cnts.size(); ++i) {
        dim_cnts[i] = (uint64_t)((double) (dim_max[i] - dim_min[i]) / interval) + 1;
    }

    // cumprod of the dimensions
    auto cumprod_dim = std::vector<ssize_t>(n_dim);
    cumprod_dim[0] = 1;
    for (int i = 0; i < dim_cnts.size() - 1; ++i) {
        cumprod_dim[i + 1] = cumprod_dim[i] * dim_cnts[i];
    }

    // compute num grids
    uint64_t num_grids = cumprod_dim.back() * dim_cnts.back();

    // cache the sum of counts and volumes, etc.
    auto count_cache = std::vector<double>(num_grids);

    auto volume_cache = std::vector<double>(num_grids);

    auto non_adjust_cache = std::vector<double>(num_grids * (n_dim - 1));

    // this variable will store the non adjust dimensions because they are prepared while the flat_idx is being computed.
    // Therefore it is not possible to directly store the non_adjust dimension to the flat_idx.
    std::vector<double> non_adjust_tmp(n_dim - 1);

    // compute the assignments
    ssize_t num_effective_rows = 0;
    for (int i = 0; i < n_rows; ++i) {
        auto count = ptr[(i + 1) * n_col - 1];
        ssize_t flat_idx = 0;
        double volume = 1.;// no need to use kshape as it will be divided later to compute the volume average.
        for (int j = 0; j < n_dim; ++j) {
            auto idx_dim = i * n_dim + j;

            // sz_partition is not the actual size because it may have been log10-ed.
            double sz_partition = sz_array[idx_dim];
            flat_idx += (ssize_t)((sz_partition - dim_min[j]) / interval) * cumprod_dim[j];

            // sz_actual is the actual size from the original array
            auto idx_actual = i * n_col + j;
            double sz_actual = ptr[idx_actual];
            volume *= pow(sz_actual, ws_ptr[j]);

            // if there is non-adjusting dimension, store them to compute mean value.
            if (j > 0) {
                non_adjust_tmp[j - 1] = sz_actual * count;
            }
        }
        // flat_idx is ready to access.

        if (count_cache[flat_idx] == 0) {
            num_effective_rows++;// remember how many unique effective rows are there to create the returning array.
        }
        count_cache[flat_idx] += count;
        volume_cache[flat_idx] += volume * count;

        // store the non adjusting dimensions after the flat_idx is ready..
        for (int j = 1; j < n_dim; ++j) {
            non_adjust_cache[flat_idx * j] += non_adjust_tmp[j - 1];
        }
    }

    // create the returning array
    const uint64_t return_array_size = n_col * num_effective_rows;
    auto ret_array = std::vector<double>(return_array_size);

    // restore the returnin grid
    auto ret_idx = 0;
    for (int i = 0; i < num_grids; ++i) {
        auto cnt = count_cache[i];
        if (cnt == 0.) {
            // this grid row is not effective
            continue;
        }

        // set the count
        auto count_idx = n_col * (ret_idx + 1) - 1;
        ret_array[count_idx] = cnt;

        // set the non-adjusting dimension
        auto vol = volume_cache[i];
        auto mean_vol = vol / cnt;
        auto non_adjusting_base_idx = n_col * ret_idx + 1;

        if (n_dim > 1) {
            for (int j = 1; j < n_dim; ++j) {
                auto non_adj_data = non_adjust_cache[i * j];
                auto non_adj_mean = non_adj_data / cnt;
                ret_array[non_adjusting_base_idx + j - 1] = non_adj_mean;
                mean_vol /= pow(non_adj_mean, ws_ptr[j]);
            }
        }
        auto adjust_dim_equivalent_size = pow(mean_vol, 1. / ws_ptr[0]);

        // set the adjusting dimension
        auto adjusting_idx = n_col * ret_idx;
        ret_array[adjusting_idx] = adjust_dim_equivalent_size;

        ret_idx++;
    }

    return py::array_t<double>(
        {num_effective_rows, n_col},// shape
        ret_array.data());          // numpy array references this parent
}

PYBIND11_MODULE(compress_binding, m) {
    m.doc() = "compress native plugin";

    m.def("compress", &compress, "A function which adds two numbers", py::return_value_policy::move);
}
