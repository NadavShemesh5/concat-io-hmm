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
    py::array_t<double> transmat_batch_,
    py::array_t<double> frameprob_batch_,
    py::array_t<double> startprob_batch_,
    py::array_t<int> lengths_
) {
    auto transmat = transmat_batch_.unchecked<4>();
    auto frameprob = frameprob_batch_.unchecked<3>();
    auto startprob = startprob_batch_.unchecked<2>();
    auto lengths = lengths_.unchecked<1>();

    auto N = frameprob.shape(0);
    auto T_max = frameprob.shape(1);
    auto S = frameprob.shape(2);

    auto fwdlattice_ = py::array_t<double>{{N, T_max, S}};
    auto fwd = fwdlattice_.mutable_unchecked<3>();

    auto scaling_ = py::array_t<double>{{N, T_max}};
    auto scaling = scaling_.mutable_unchecked<2>();

    double total_log_prob = 0.0;

    py::gil_scoped_release nogil;
    for (ssize_t n = 0; n < N; ++n) {
        auto T_n = lengths(n);
        double log_prob_n = 0.0;

        double sum = 0.0;
        for (ssize_t i = 0; i < S; ++i) {
            double p = startprob(n, i) * frameprob(n, 0, i);
            fwd(n, 0, i) = p;
            sum += p;
        }

        double scale = 1.0 / sum;
        scaling(n, 0) = scale;
        log_prob_n -= std::log(scale);
        for (ssize_t i = 0; i < S; ++i) fwd(n, 0, i) *= scale;
        for (ssize_t t = 1; t < T_n; ++t) {
            sum = 0.0;
            for (ssize_t j = 0; j < S; ++j) {
                double acc = 0.0;
                for (ssize_t i = 0; i < S; ++i) {
                    acc += fwd(n, t - 1, i) * transmat(n, t, i, j);
                }
                double val = acc * frameprob(n, t, j);
                fwd(n, t, j) = val;
                sum += val;
            }

            scale = 1.0 / sum;
            scaling(n, t) = scale;
            log_prob_n -= std::log(scale);
            for (ssize_t j = 0; j < S; ++j) fwd(n, t, j) *= scale;
        }

        for (ssize_t t = T_n; t < T_max; ++t) {
            for (ssize_t s = 0; s < S; ++s) fwd(n, t, s) = 0.0;
            scaling(n, t) = 1.0;
        }
        total_log_prob += log_prob_n;
    }

    return {total_log_prob, fwdlattice_, scaling_};
}

py::array_t<double> backward(
    py::array_t<double> transmat_batch_,
    py::array_t<double> frameprob_batch_,
    py::array_t<double> scaling_batch_,
    py::array_t<int> lengths_
) {
    auto transmat = transmat_batch_.unchecked<4>();
    auto frameprob = frameprob_batch_.unchecked<3>();
    auto scaling = scaling_batch_.unchecked<2>();
    auto lengths = lengths_.unchecked<1>();

    auto N = frameprob.shape(0);
    auto T_max = frameprob.shape(1);
    auto S = frameprob.shape(2);

    auto bwdlattice_ = py::array_t<double>{{N, T_max, S}};
    auto bwd = bwdlattice_.mutable_unchecked<3>();

    py::gil_scoped_release nogil;
    std::fill_n(bwd.mutable_data(0, 0, 0), bwd.size(), 0.0);
    for (ssize_t n = 0; n < N; ++n) {
        auto T_n = lengths(n);
        if (T_n == 0) continue;
        for (ssize_t i = 0; i < S; ++i) bwd(n, T_n - 1, i) = scaling(n, T_n - 1);
        for (ssize_t t = T_n - 2; t >= 0; --t) {
            for (ssize_t i = 0; i < S; ++i) {
                double acc = 0.0;
                for (ssize_t j = 0; j < S; ++j) {
                    acc += transmat(n, t + 1, i, j) * frameprob(n, t + 1, j) * bwd(n, t + 1, j);
                }
                bwd(n, t, i) = acc * scaling(n, t);
            }
        }
    }
    return bwdlattice_;
}

void compute_xi_sum(
    py::array_t<double> fwd_batch_,
    py::array_t<double> transmat_batch_,
    py::array_t<double> bwd_batch_,
    py::array_t<double> frameprob_batch_,
    py::array_t<int> inputs_batch_,
    py::array_t<int> lengths_,
    py::array_t<double> out_stats_
) {
    auto fwd = fwd_batch_.unchecked<3>();
    auto transmat = transmat_batch_.unchecked<4>();
    auto bwd = bwd_batch_.unchecked<3>();
    auto frameprob = frameprob_batch_.unchecked<3>();
    auto inputs = inputs_batch_.unchecked<2>();
    auto lengths = lengths_.unchecked<1>();
    auto out = out_stats_.mutable_unchecked<3>();

    auto N = frameprob.shape(0);
    auto S = frameprob.shape(2);

    py::gil_scoped_release nogil;

    for (ssize_t n = 0; n < N; ++n) {
        auto T_n = lengths(n);
        for (ssize_t t = 0; t < T_n - 1; ++t) {
            auto u_next = inputs(n, t + 1);
            for (ssize_t i = 0; i < S; ++i) {
                for (ssize_t j = 0; j < S; ++j) {
                    double val = fwd(n, t, i) * transmat(n, t + 1, i, j) * frameprob(n, t + 1, j) * bwd(n, t + 1, j);
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
