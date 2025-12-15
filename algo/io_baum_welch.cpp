#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace py = pybind11;
using ssize_t = Py_ssize_t;

const double MIN_SUM = 1e-300;

std::tuple<double, py::array_t<double>, py::array_t<double>> forward(
    py::array_t<double> trans_mat_,
    py::array_t<double> emit_mat_,
    py::array_t<double> start_mat_,
    py::array_t<int> inputs_,
    py::array_t<int> outputs_,
    py::array_t<int> lengths_
) {
    auto trans_mat = trans_mat_.unchecked<3>();
    auto emit_mat = emit_mat_.unchecked<3>();
    auto start_mat = start_mat_.unchecked<2>();
    auto inputs = inputs_.unchecked<2>();
    auto outputs = outputs_.unchecked<2>();
    auto lengths = lengths_.unchecked<1>();

    auto N = inputs.shape(0);
    auto T_max = inputs.shape(1);
    auto S = trans_mat.shape(1);

    auto fwdlattice_ = py::array_t<double>{{N, T_max, S}};
    auto fwd = fwdlattice_.mutable_unchecked<3>();

    auto scaling_ = py::array_t<double>{{N, T_max}};
    auto scaling = scaling_.mutable_unchecked<2>();

    double total_log_prob = 0.0;

    py::gil_scoped_release nogil;
    for (ssize_t n = 0; n < N; ++n) {
        auto T_n = lengths(n);
        double log_prob_n = 0.0;
        auto u_0 = inputs(n, 0);
        auto v_0 = outputs(n, 0);
        double sum = 0.0;
        for (ssize_t i = 0; i < S; ++i) {
            double p = start_mat(i, u_0) * emit_mat(u_0, i, v_0);
            fwd(n, 0, i) = p;
            sum += p;
        }

        if (sum < MIN_SUM) sum = 1.0;
        double scale = 1.0 / sum;
        scaling(n, 0) = scale;
        log_prob_n -= std::log(scale);
        for (ssize_t i = 0; i < S; ++i) fwd(n, 0, i) *= scale;
        for (ssize_t t = 1; t < T_n; ++t) {
            auto u_t = inputs(n, t);
            auto v_t = outputs(n, t);

            sum = 0.0;
            for (ssize_t j = 0; j < S; ++j) {
                double acc = 0.0;
                for (ssize_t i = 0; i < S; ++i) {
                    acc += fwd(n, t - 1, i) * trans_mat(u_t, i, j);
                }

                double val = acc * emit_mat(u_t, j, v_t);
                fwd(n, t, j) = val;
                sum += val;
            }
            if (sum < MIN_SUM) sum = 1.0;

            scale = 1.0 / sum;
            scaling(n, t) = scale;
            log_prob_n -= std::log(scale);
            for (ssize_t j = 0; j < S; ++j) fwd(n, t, j) *= scale;
        }
        total_log_prob += log_prob_n;
    }

    return {total_log_prob, fwdlattice_, scaling_};
}

py::array_t<double> backward(
    py::array_t<double> trans_mat_,
    py::array_t<double> emit_mat_,
    py::array_t<double> scaling_batch_,
    py::array_t<int> inputs_,
    py::array_t<int> outputs_,
    py::array_t<int> lengths_
) {
    auto trans_mat = trans_mat_.unchecked<3>();
    auto emit_mat = emit_mat_.unchecked<3>();
    auto scaling = scaling_batch_.unchecked<2>();
    auto inputs = inputs_.unchecked<2>();
    auto outputs = outputs_.unchecked<2>();
    auto lengths = lengths_.unchecked<1>();

    auto N = inputs.shape(0);
    auto T_max = inputs.shape(1);
    auto S = trans_mat.shape(1);

    auto bwdlattice_ = py::array_t<double>{{N, T_max, S}};
    auto bwd = bwdlattice_.mutable_unchecked<3>();

    py::gil_scoped_release nogil;

    std::fill_n(bwd.mutable_data(0, 0, 0), bwd.size(), 0.0);
    for (ssize_t n = 0; n < N; ++n) {
        auto T_n = lengths(n);
        if (T_n == 0) continue;
        for (ssize_t i = 0; i < S; ++i) bwd(n, T_n - 1, i) = scaling(n, T_n - 1);
        for (ssize_t t = T_n - 2; t >= 0; --t) {
            auto u_next = inputs(n, t + 1);
            auto v_next = outputs(n, t + 1);

            for (ssize_t i = 0; i < S; ++i) {
                double acc = 0.0;
                for (ssize_t j = 0; j < S; ++j) {
                    acc += trans_mat(u_next, i, j) * emit_mat(u_next, j, v_next) * bwd(n, t + 1, j);
                }
                bwd(n, t, i) = acc * scaling(n, t);
            }
        }
    }
    return bwdlattice_;
}

void compute_xi_sum(
    py::array_t<double> fwd_batch_,
    py::array_t<double> trans_mat_,
    py::array_t<double> bwd_batch_,
    py::array_t<double> emit_mat_,
    py::array_t<int> inputs_batch_,
    py::array_t<int> outputs_batch_,
    py::array_t<int> lengths_,
    py::array_t<double> out_stats_
) {
    auto fwd = fwd_batch_.unchecked<3>();
    auto trans_mat = trans_mat_.unchecked<3>();
    auto bwd = bwd_batch_.unchecked<3>();
    auto emit_mat = emit_mat_.unchecked<3>();
    auto inputs = inputs_batch_.unchecked<2>();
    auto outputs = outputs_batch_.unchecked<2>();
    auto lengths = lengths_.unchecked<1>();
    auto out = out_stats_.mutable_unchecked<3>();

    auto N = inputs.shape(0);
    auto S = trans_mat.shape(1);

    py::gil_scoped_release nogil;
    for (ssize_t n = 0; n < N; ++n) {
        auto T_n = lengths(n);
        for (ssize_t t = 0; t < T_n - 1; ++t) {
            auto u_next = inputs(n, t + 1);
            auto v_next = outputs(n, t + 1);
            for (ssize_t i = 0; i < S; ++i) {
                for (ssize_t j = 0; j < S; ++j) {
                    double val = fwd(n, t, i)
                               * trans_mat(u_next, i, j)
                               * emit_mat(u_next, j, v_next)
                               * bwd(n, t + 1, j);

                    out(u_next, i, j) += val;
                }
            }
        }
    }
}

PYBIND11_MODULE(io_baum_welch, m) {
  m.def("forward", forward)
   .def("backward", backward)
   .def("compute_xi_sum", compute_xi_sum);
}