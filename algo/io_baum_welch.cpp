#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cfenv>
#include <limits>

namespace py = pybind11;
using ssize_t = Py_ssize_t;

std::tuple<double, py::array_t<double>, py::array_t<double>> forward(
  py::array_t<double> transmat_,
  py::array_t<double> frameprob_)
{
  auto min_sum = 1e-300;
  auto transmat = transmat_.unchecked<3>();
  auto frameprob = frameprob_.unchecked<2>();

  auto ns = frameprob.shape(0);
  auto nc = frameprob.shape(1);

  auto fwdlattice_ = py::array_t<double>{{ns, nc}};
  auto fwd = fwdlattice_.mutable_unchecked<2>();
  auto scaling_ = py::array_t<double>{{ns}};
  auto scaling = scaling_.mutable_unchecked<1>();
  auto log_prob = 0.;
  {
    py::gil_scoped_release nogil;
    std::fill_n(fwd.mutable_data(0, 0), fwd.size(), 0);
    for (auto i = 0; i < nc; ++i) fwd(0, i) = frameprob(0, i);
    auto sum = std::accumulate(&fwd(0, 0), &fwd(0, nc), 0.);
    if (sum < min_sum) throw std::range_error{"forward underflow"};

    auto scale = scaling(0) = 1. / sum;
    log_prob -= std::log(scale);
    for (auto i = 0; i < nc; ++i) fwd(0, i) *= scale;
    for (auto t = 1; t < ns; ++t) {
      for (auto j = 0; j < nc; ++j) {
        double acc = 0;
        for (auto i = 0; i < nc; ++i) {
          acc += fwd(t - 1, i) * transmat(t, i, j);
        }
        fwd(t, j) = acc * frameprob(t, j);
      }

      sum = std::accumulate(&fwd(t, 0), &fwd(t, nc), 0.);
      if (sum < min_sum) throw std::range_error{"forward underflow"};

      scale = scaling(t) = 1. / sum;
      log_prob -= std::log(scale);
      for (auto j = 0; j < nc; ++j) fwd(t, j) *= scale;
    }
  }
  return {log_prob, fwdlattice_, scaling_};
}

py::array_t<double> backward(
  py::array_t<double> transmat_,
  py::array_t<double> frameprob_,
  py::array_t<double> scaling_)
{
  auto transmat = transmat_.unchecked<3>();
  auto frameprob = frameprob_.unchecked<2>();
  auto scaling = scaling_.unchecked<1>();

  auto ns = frameprob.shape(0);
  auto nc = frameprob.shape(1);

  auto bwdlattice_ = py::array_t<double>{{ns, nc}};
  auto bwd = bwdlattice_.mutable_unchecked<2>();

  py::gil_scoped_release nogil;
  std::fill_n(bwd.mutable_data(0, 0), bwd.size(), 0);

  for (auto i = 0; i < nc; ++i) bwd(ns - 1, i) = scaling(ns - 1);

  for (auto t = ns - 2; t >= 0; --t) {
    double scale_t = scaling(t);
    for (auto i = 0; i < nc; ++i) {
      double acc = 0;
      for (auto j = 0; j < nc; ++j) {
        acc += transmat(t + 1, i, j) * frameprob(t + 1, j) * bwd(t + 1, j);
      }
      bwd(t, i) = acc * scale_t;
    }
  }
  return bwdlattice_;
}

void accumulate_trans(
    py::array_t<double> fwd_,
    py::array_t<double> bwd_,
    py::array_t<double> trans_static_,
    py::array_t<double> frameprob_,
    py::object input_obj,
    py::array_t<double> out_counts_)
{
    auto fwd = fwd_.unchecked<2>();
    auto bwd = bwd_.unchecked<2>();
    auto trans = trans_static_.unchecked<3>();
    auto frameprob = frameprob_.unchecked<2>();
    auto out = out_counts_.mutable_unchecked<3>();

    auto T = frameprob.shape(0);
    auto S = frameprob.shape(1);
    auto K = trans.shape(0);

    bool is_hard_in = (py::isinstance<py::array_t<int64_t>>(input_obj) && input_obj.cast<py::array_t<int64_t>>().ndim() == 1);

    py::gil_scoped_release nogil;
    if (is_hard_in) {
        auto input = input_obj.cast<py::array_t<int64_t>>().unchecked<1>();

        for (auto t = 0; t < T - 1; ++t) {
            ssize_t k = (ssize_t)input(t + 1);
            if (k < 0 || k >= K) continue;

            for (auto i = 0; i < S; ++i) {
                double f_i = fwd(t, i);
                if (f_i <= 1e-12) continue;
                for (auto j = 0; j < S; ++j) {
                     out(k, i, j) += f_i * trans(k, i, j) * frameprob(t + 1, j) * bwd(t + 1, j);
                }
            }
        }
    }
    else {
        auto input = input_obj.cast<py::array_t<double>>().unchecked<2>();

        for (auto t = 0; t < T - 1; ++t) {
            for (auto k = 0; k < K; ++k) {
                double w_k = input(t + 1, k);
                if (w_k <= 1e-12) continue;

                for (auto i = 0; i < S; ++i) {
                    double f_i = fwd(t, i);
                    if (f_i <= 1e-12) continue;
                    for (auto j = 0; j < S; ++j) {
                        double val = f_i * trans(k, i, j) * frameprob(t + 1, j) * bwd(t + 1, j);
                        out(k, i, j) += val * w_k;
                    }
                }
            }
        }
    }
}

void accumulate_obs(
    py::array_t<double> posteriors_,
    py::object input_obj,
    py::object output_obj,
    py::array_t<double> out_counts_)
{
    auto gamma = posteriors_.unchecked<2>();
    auto out = out_counts_.mutable_unchecked<3>(); // Fix: Mutable

    auto T = gamma.shape(0);
    auto S = gamma.shape(1);
    auto K = out.shape(0);
    auto Y = out.shape(2);

    bool is_hard_in = (py::isinstance<py::array_t<int64_t>>(input_obj) && input_obj.cast<py::array_t<int64_t>>().ndim() == 1);
    bool is_hard_out = (py::isinstance<py::array_t<int64_t>>(output_obj) && output_obj.cast<py::array_t<int64_t>>().ndim() == 1);

    py::gil_scoped_release nogil;
    if (is_hard_in && is_hard_out) {
        auto in = input_obj.cast<py::array_t<int64_t>>().unchecked<1>();
        auto ot = output_obj.cast<py::array_t<int64_t>>().unchecked<1>();

        for (auto t = 0; t < T; ++t) {
            ssize_t k = in(t);
            ssize_t y = ot(t);
            if (k >= 0 && k < K && y >= 0 && y < Y) {
                for (auto i = 0; i < S; ++i) out(k, i, y) += gamma(t, i);
            }
        }
    }
    else if (is_hard_in && !is_hard_out) {
        auto in = input_obj.cast<py::array_t<int64_t>>().unchecked<1>();
        auto ot = output_obj.cast<py::array_t<double>>().unchecked<2>();

        for (auto t = 0; t < T; ++t) {
            ssize_t k = in(t);
            if (k < 0 || k >= K) continue;

            for (auto y = 0; y < Y; ++y) {
                double w_y = ot(t, y);
                if (w_y <= 1e-12) continue;
                for (auto i = 0; i < S; ++i) out(k, i, y) += gamma(t, i) * w_y;
            }
        }
    }
    else if (!is_hard_in && is_hard_out) {
        auto in = input_obj.cast<py::array_t<double>>().unchecked<2>();
        auto ot = output_obj.cast<py::array_t<int64_t>>().unchecked<1>();

        for (auto t = 0; t < T; ++t) {
            ssize_t y = ot(t);
            if (y < 0 || y >= Y) continue;

            for (auto k = 0; k < K; ++k) {
                double w_k = in(t, k);
                if (w_k <= 1e-12) continue;
                for (auto i = 0; i < S; ++i) out(k, i, y) += gamma(t, i) * w_k;
            }
        }
    }
    else {
        auto in = input_obj.cast<py::array_t<double>>().unchecked<2>();
        auto ot = output_obj.cast<py::array_t<double>>().unchecked<2>();

        for (auto t = 0; t < T; ++t) {
            for (auto k = 0; k < K; ++k) {
                double w_k = in(t, k);
                if (w_k <= 1e-12) continue;

                for (auto y = 0; y < Y; ++y) {
                    double w_y = ot(t, y);
                    if (w_y <= 1e-12) continue;

                    double weight = w_k * w_y;
                    for (auto i = 0; i < S; ++i) out(k, i, y) += gamma(t, i) * weight;
                }
            }
        }
    }
}

PYBIND11_MODULE(io_baum_welch, m) {
  m.def("forward", &forward);
  m.def("backward", &backward);
  m.def("accumulate_trans", &accumulate_trans);
  m.def("accumulate_obs", &accumulate_obs);
}